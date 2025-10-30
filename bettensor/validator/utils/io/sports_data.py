import os
import json
import time
from typing import List, Dict, Any
import uuid
import sqlite3
import requests
import bittensor as bt
from dateutil import parser
from .sports_config import sports_config
from datetime import datetime, timedelta, timezone
from ..scoring.entropy_system import EntropySystem
from .bettensor_api_client import BettensorAPIClient
from bettensor.validator.utils.database.database_manager import DatabaseManager
import traceback
import asyncio
import async_timeout
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ColumnElement
from sqlalchemy import text


class SportsData:
    """
    SportsData class is responsible for fetching and updating sports data from either BettensorAPI or external API.
    """

    # Constants for chunking and timeouts
    TRANSACTION_TIMEOUT = 20  # Reduced to give more headroom for validator timeout
    CHUNK_SIZE = 5  # Smaller chunks for more granular processing
    ENTROPY_BATCH_SIZE = 2  # Very small batches for entropy updates
    MAX_RETRIES = 3
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        entropy_system: EntropySystem,
        api_client,
        netuid: int,
    ):
        self.db_manager = db_manager
        self.entropy_system = entropy_system
        self.api_client = api_client
        self.netuid = netuid
        self.all_games = []
        self._last_processed_time = None

    async def fetch_and_update_game_data(self, last_api_call):
        """Fetch and update game data with proper transaction and timeout handling"""
        try:
            # Check if we've processed this time period recently
            current_time = datetime.now(timezone.utc)
            if (self._last_processed_time and 
                (current_time - self._last_processed_time) < timedelta(minutes=5)):
                bt.logging.info("Skipping update - last update was less than 5 minutes ago")
                return []

            # Get all games that need updates
            all_games = await self.api_client.fetch_all_game_data(last_api_call)
            bt.logging.info(f"Fetched {len(all_games)} games from API")
            
            if not all_games:
                self._last_processed_time = current_time
                return []

            # Filter out invalid games
            valid_games = [game for game in all_games if isinstance(game, dict) and 'externalId' in game]
            bt.logging.info(f"Found {len(valid_games)} valid games")

            # Group games by date
            games_by_date = self._group_games_by_date(valid_games)
            if not games_by_date:
                self._last_processed_time = current_time
                return []

            # Process games in date order with independent error handling
            inserted_ids = []
            entropy_updates_needed = []
            
            for date, date_games in sorted(games_by_date.items()):
                bt.logging.info(f"Processing {len(date_games)} games for date {date}")
                
                # Process database updates for this date
                date_ids = await self._process_date_games(date_games)
                if date_ids:
                    inserted_ids.extend(date_ids)
                    entropy_updates_needed.extend([g for g in date_games if g.get("externalId") in date_ids])
            
            # Process entropy updates in small batches with independent timeouts
            if entropy_updates_needed:
                await self._process_entropy_updates_in_batches(entropy_updates_needed)
            
            # Fix any games with null create_date or game_id
            await self._fix_null_game_data()
            
            # Fix any miners with null hotkey/coldkey values
            await self._fix_null_miner_keys()
            
            # Update last processed time AFTER all processing steps are done
            # This ensures the cooldown applies even if no games were inserted/updated
            # but the API call and processing steps were completed.
            self._last_processed_time = current_time
                        
            self.all_games = valid_games
            return inserted_ids
                
        except Exception as e:
            bt.logging.error(f"Error in game data update: {str(e)}")
            bt.logging.error(traceback.format_exc())
            # Do not update _last_processed_time on error to allow immediate retry
            return []

    def _cleanup_processed_ids(self):
        """Clean up processed game IDs older than 24 hours to prevent memory growth"""
        try:
            if len(self._processed_game_ids) > 10000:  # Arbitrary limit
                bt.logging.info("Cleaning up processed game IDs cache")
                self._processed_game_ids.clear()
        except Exception as e:
            bt.logging.error(f"Error cleaning up processed IDs: {str(e)}")

    def _group_games_by_date(self, games):
        """Group games by date with validation"""
        games_by_date = {}
        for game in games:
            try:
                if not isinstance(game, dict) or 'date' not in game:
                    bt.logging.warning(f"Skipping invalid game: {game}")
                    continue
                    
                date = datetime.fromisoformat(game['date'].replace('Z', '+00:00')).date().isoformat()
                games_by_date.setdefault(date, []).append(game)
            except Exception as e:
                bt.logging.error(f"Error processing game: {e}, Game data: {game}")
                continue
        
        bt.logging.info(f"Grouped {sum(len(games) for games in games_by_date.values())} valid games into {len(games_by_date)} dates")
        return games_by_date

    async def _process_date_games(self, date_games):
        """Process games for a specific date, handling insertion or update via UPSERT."""
        inserted_ids = []
        try:
            bt.logging.info(f"Processing {len(date_games)} games received from API for date")

            # Filter games for basic validity (must have externalId)
            valid_games = [game for game in date_games if isinstance(game, dict) and game.get("externalId")]
            bt.logging.info(f"Processing {len(valid_games)} valid games after filtering")

            # Process valid games in chunks using UPSERT logic
            for i in range(0, len(valid_games), self.CHUNK_SIZE):
                chunk = valid_games[i:i + self.CHUNK_SIZE]
                # _process_game_chunk_with_retries will call insert_or_update_games,
                # which uses the UPSERT query, eliminating the need for pre-fetching.
                chunk_ids = await self._process_game_chunk_with_retries(chunk)
                if chunk_ids:
                    inserted_ids.extend(chunk_ids)
                    
        except Exception as e:
            # Adjusted log message for clarity
            bt.logging.error(f"Error processing games batch for date: {str(e)}") 
            bt.logging.error(traceback.format_exc())
            
        return inserted_ids

    async def _process_game_chunk_with_retries(self, chunk: List[Dict[str, Any]], max_retries: int = 3) -> List[str]:
        """Process a chunk of games with retries and transaction handling."""
        for attempt in range(max_retries):
            try:
                bt.logging.debug(f"Starting game chunk processing (attempt {attempt + 1}/{max_retries})")
                bt.logging.debug(f"Chunk size: {len(chunk)} games")
                
                # Log the first game in the chunk for debugging
                if chunk:
                    bt.logging.debug(f"Sample game data: {chunk[0]}")
                
                async with async_timeout.timeout(self.TRANSACTION_TIMEOUT):
                    bt.logging.debug(f"Transaction timeout set to {self.TRANSACTION_TIMEOUT} seconds")
                    async with self.db_manager.transaction() as session:
                        bt.logging.debug("Transaction started")
                        chunk_ids = await self.insert_or_update_games(chunk, session)
                        bt.logging.debug(f"Transaction completed successfully. Processed {len(chunk_ids)} games")
                        return chunk_ids
                        
            except asyncio.TimeoutError:
                bt.logging.error(f"Transaction timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    bt.logging.error("Max retries reached for transaction timeout")
                    raise
                await asyncio.sleep(1)
                
            except asyncio.CancelledError as e:
                bt.logging.error(f"Transaction cancelled on attempt {attempt + 1}")
                bt.logging.error(f"Cancellation context: {e.__context__}")
                bt.logging.error(f"Cancellation traceback: {traceback.format_exc()}")
                if attempt == max_retries - 1:
                    bt.logging.error("Max retries reached for transaction cancellation")
                    raise
                await asyncio.sleep(1)
                
            except Exception as e:
                bt.logging.error(f"Error processing game chunk on attempt {attempt + 1}: {str(e)}")
                bt.logging.error(f"Error traceback: {traceback.format_exc()}")
                if attempt == max_retries - 1:
                    bt.logging.error("Max retries reached for general error")
                    raise
                await asyncio.sleep(1)

    async def _process_entropy_updates_in_batches(self, games):
        """Process entropy updates in very small batches, saving state once at the end."""
        processed_count = 0
        total_batches = (len(games) + self.ENTROPY_BATCH_SIZE - 1) // self.ENTROPY_BATCH_SIZE
        
        for i in range(0, len(games), self.ENTROPY_BATCH_SIZE):
            batch = games[i:i + self.ENTROPY_BATCH_SIZE]
            batch_num = i // self.ENTROPY_BATCH_SIZE + 1
            
            # Update entropy system (memory operations)
            try:
                game_data = self.prepare_game_data_for_entropy(batch)
                if not game_data:
                    continue
                
                for game in game_data:
                    await self.entropy_system.add_new_game(
                        game["id"], 
                        len(game["current_odds"]), 
                        game["current_odds"]
                    )
                processed_count += len(game_data)
                bt.logging.debug(f"Added batch {batch_num}/{total_batches} ({len(game_data)} games) to entropy system")
            except Exception as e:
                bt.logging.error(f"Error updating entropy system for batch {batch_num}/{total_batches}: {str(e)}")
                continue
        
        # Save state once after processing all batches
        if processed_count > 0:
            bt.logging.info(f"Attempting to save entropy state after processing {processed_count} games")
            retries = 0
            while retries < self.MAX_RETRIES:
                try:
                    # Use a separate long-running session for entropy state save
                    async with self.db_manager.get_long_running_session() as session:
                        await self.entropy_system.save_state()
                        bt.logging.info(f"Successfully saved entropy state.")
                        break
                except Exception as e:
                    bt.logging.error(f"Error saving entropy state (attempt {retries + 1}/{self.MAX_RETRIES}): {str(e)}")
                    retries += 1
                    if retries == self.MAX_RETRIES:
                        bt.logging.error("Failed to save entropy state after max retries")
                    await asyncio.sleep(1) # Wait before retrying
        else:
            bt.logging.info("No games processed for entropy, skipping state save.")

    async def insert_or_update_games(self, games, session):
        """Insert or update games in the database using the provided session"""
        try:
            inserted_ids = []
            
            # Filter out string entries and ensure we have a list of games
            if isinstance(games, dict):
                games = [games]
            
            valid_games = [g for g in games if isinstance(g, dict)]
            bt.logging.info(f"Inserting/updating {len(valid_games)} games")
            
            for game in valid_games:
                try:
                    external_id = str(game.get("externalId"))
                    if not external_id:
                        continue

                    # Process game data and execute update
                    params = await self._prepare_game_params(game)
                    if not params:
                        continue

                    await session.execute(
                        text(self._get_upsert_query()),
                        params
                    )
                    bt.logging.debug(f"Inserted/Updated game with external_id: {external_id}")
                    inserted_ids.append(external_id)

                except asyncio.CancelledError:
                    bt.logging.warning(f"Game processing was cancelled for external_id: {external_id}")
                    raise
                except Exception as e:
                    bt.logging.error(f"Error processing game with external_id {external_id}: {str(e)}")
                    continue

            return inserted_ids
            
        except asyncio.CancelledError:
            bt.logging.warning("Game processing was cancelled")
            raise
        except Exception as e:
            bt.logging.error(f"Error in insert_or_update_games: {str(e)}")
            raise

    async def _prepare_game_params(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare game parameters for database insertion."""
        try:
            # Map API field names to database field names
            field_mapping = {
                'externalId': 'external_id',
                'teamA': 'team_a',
                'teamB': 'team_b',
                'teamAOdds': 'team_a_odds',
                'teamBOdds': 'team_b_odds',
                'drawOdds': 'tie_odds',
                'canDraw': 'can_tie',
                'date': 'event_start_date'  # Handle both date and event_start_date
            }

            # Get event_start_date from either field
            event_start_date = game.get('event_start_date') or game.get('date')
            if not event_start_date:
                bt.logging.error(f"Missing event_start_date/date in game data: {game}")
                raise ValueError("Missing event start date in game data")

            # Parse event_start_date if it's a string
            if isinstance(event_start_date, str):
                try:
                    event_start_date = datetime.fromisoformat(event_start_date.replace('Z', '+00:00'))
                except ValueError as e:
                    bt.logging.error(f"Failed to parse event_start_date '{event_start_date}': {str(e)}")
                    raise

            # Calculate create_date as one week before event_start_date if not provided
            create_date = game.get('create_date')
            if not create_date:
                try:
                    create_date = event_start_date - timedelta(days=7)
                    bt.logging.debug(f"Setting create_date to one week before event_start_date: {create_date}")
                except Exception as e:
                    bt.logging.error(f"Error calculating create_date: {str(e)}")
                    bt.logging.error(f"event_start_date: {event_start_date}")
                    bt.logging.error(f"event_start_date type: {type(event_start_date)}")
                    raise

            # Parse create_date if it's a string
            if isinstance(create_date, str):
                try:
                    create_date = datetime.fromisoformat(create_date.replace('Z', '+00:00'))
                except ValueError as e:
                    bt.logging.error(f"Failed to parse create_date '{create_date}': {str(e)}")
                    raise

            # Ensure last_update_date is set
            last_update_date = game.get('last_update_date')
            if not last_update_date:
                last_update_date = datetime.now(timezone.utc)
                bt.logging.debug(f"Setting last_update_date to current time: {last_update_date}")

            # Parse last_update_date if it's a string
            if isinstance(last_update_date, str):
                try:
                    last_update_date = datetime.fromisoformat(last_update_date.replace('Z', '+00:00'))
                except ValueError as e:
                    bt.logging.error(f"Failed to parse last_update_date '{last_update_date}': {str(e)}")
                    raise

            # Map API fields to database fields with proper type handling
            params = {
                'external_id': str(game.get(field_mapping.get('externalId', 'external_id'), game.get('externalId', ''))),
                'event_start_date': event_start_date,
                'team_a': str(game.get(field_mapping.get('teamA', 'team_a'), game.get('teamA', ''))),
                'team_b': str(game.get(field_mapping.get('teamB', 'team_b'), game.get('teamB', ''))),
                'sport': str(game.get('sport', '')),
                'league': str(game.get('league', '')),
                'active': bool(game.get('active', True)),
                'outcome': self._process_game_outcome(game),
                'team_a_odds': float(game.get(field_mapping.get('teamAOdds', 'team_a_odds'), game.get('teamAOdds', 0.0))),
                'team_b_odds': float(game.get(field_mapping.get('teamBOdds', 'team_b_odds'), game.get('teamBOdds', 0.0))),
                'tie_odds': float(game.get(field_mapping.get('drawOdds', 'tie_odds'), game.get('drawOdds', 0.0))) if game.get('drawOdds') is not None else None,
                'can_tie': bool(game.get(field_mapping.get('canDraw', 'can_tie'), game.get('canDraw', False))),
                'create_date': create_date,
                'last_update_date': last_update_date,
                'model_config': game.get('model_config', {}),
                'game_id': game.get('game_id', str(uuid.uuid4()))
            }

            # Log the prepared parameters for debugging
            bt.logging.debug(f"Prepared game parameters: {params}")
            return params

        except Exception as e:
            bt.logging.error(f"Error preparing game parameters: {str(e)}")
            bt.logging.error(f"Game data: {game}")
            bt.logging.error(f"Error type: {type(e)}")
            bt.logging.error(f"Error traceback: {traceback.format_exc()}")
            raise

    def _process_game_outcome(self, game):
        """Process game outcome with validation"""
        outcome = game.get("outcome")
        if outcome is None or outcome == "Unfinished":
            return 3
        elif isinstance(outcome, str):
            if outcome == "TeamAWin":
                return 0
            elif outcome == "TeamBWin":
                return 1
            elif outcome == "Draw":
                return 2
            else:
                bt.logging.warning(f"Unexpected game outcome string received from API for game {game.get('externalId', 'Unknown_ID')}: '{outcome}'. Defaulting to Unfinished (3).")
                return 3  # Default to Unfinished for unknown strings
        else:
            bt.logging.warning(f"Unexpected game outcome type received from API for game {game.get('externalId', 'Unknown_ID')}: '{type(outcome)}'. Defaulting to Unfinished (3).")
            return 3  # Default to Unfinished for unknown types

    def _get_upsert_query(self):
        """Get the SQL query for upserting game data"""
        return """
        INSERT INTO game_data (
            game_id, team_a, team_b, sport, league, external_id, create_date, 
            last_update_date, event_start_date, active, outcome, team_a_odds, 
            team_b_odds, tie_odds, can_tie
        )
        VALUES (
            :game_id, :team_a, :team_b, :sport, :league, :external_id, :create_date,
            :last_update_date, :event_start_date, :active, :outcome, :team_a_odds,
            :team_b_odds, :tie_odds, :can_tie
        )
        ON CONFLICT(external_id) DO UPDATE SET
            game_id = COALESCE(game_data.game_id, excluded.game_id),
            create_date = COALESCE(game_data.create_date, excluded.create_date),
            team_a = excluded.team_a,
            team_b = excluded.team_b,
            sport = excluded.sport,
            league = excluded.league,
            team_a_odds = excluded.team_a_odds,
            team_b_odds = excluded.team_b_odds,
            tie_odds = excluded.tie_odds,
            event_start_date = excluded.event_start_date,
            active = excluded.active,
            outcome = excluded.outcome,
            last_update_date = excluded.last_update_date,
            can_tie = excluded.can_tie
        """

    def prepare_game_data_for_entropy(self, games):
        game_data = []
        for game in games:
            try:
                # Skip if game is not a dictionary
                if not isinstance(game, dict):
                    bt.logging.debug(f"Skipping non-dict game entry: {game}")
                    continue
                    
                # Check if required fields exist
                if not all(key in game for key in ["externalId", "teamAOdds", "teamBOdds", "sport"]):
                    bt.logging.debug(f"Skipping game missing required fields: {game}")
                    continue
                    
                game_data.append({
                    "id": game["externalId"],
                    "predictions": {},  # No predictions yet for new games
                    "current_odds": [
                        float(game["teamAOdds"]),
                        float(game["teamBOdds"]),
                        float(game.get("drawOdds", 0.0)) if game['sport'] != 'Football' else 0.0
                    ],
                })
            except Exception as e:
                bt.logging.error(f"Error preparing game for entropy: {e}")
                bt.logging.debug(f"Problematic game data: {game}")
                continue
                
        return game_data

    async def update_predictions_with_payouts(self, external_ids):
        """
        Retrieve all predictions associated with the provided external IDs, determine if each prediction won,
        calculate payouts, and update the predictions in the database.

        Args:
            external_ids (List[str]): List of external_id's of the games that were inserted/updated.
        """
        try:
            if not external_ids:
                bt.logging.info("No external IDs provided for updating predictions.")
                return

            # Fetch outcomes for the given external_ids
            query = """
                SELECT external_id, outcome
                FROM game_data
                WHERE external_id IN ({seq})
            """.format(
                seq=",".join(["?"] * len(external_ids))
            )
            game_outcomes = await self.db_manager.fetch_all(query, tuple(external_ids))
            game_outcome_map = {
                external_id: outcome for external_id, outcome in game_outcomes
            }

            bt.logging.info(f"Fetched outcomes for {len(game_outcomes)} games.")

            # Fetch all predictions associated with the external_ids
            query = """
                SELECT prediction_id, miner_uid, game_id, predicted_outcome, predicted_odds, wager
                FROM predictions
                WHERE game_id IN ({seq}) 
            """.format(
                seq=",".join(["?"] * len(external_ids))
            )
            predictions = await self.db_manager.fetch_all(query, tuple(external_ids))

            bt.logging.info(f"Fetched {len(predictions)} predictions to process.")

            for prediction in predictions:
                
                (
                    prediction_id,
                    miner_uid,
                    game_id,
                    predicted_outcome,
                    predicted_odds,
                    wager,
                ) = prediction
                if game_id == "game_id":
                    continue
                actual_outcome = game_outcome_map.get(game_id)

                if actual_outcome is None:
                    bt.logging.warning(
                        f"No outcome found for game {game_id}. Skipping prediction {prediction_id}."
                    )
                    continue

                is_winner = predicted_outcome == actual_outcome
                payout = wager * predicted_odds if is_winner else 0

                update_query = """
                    UPDATE predictions
                    SET result = ?, payout = ?, processed = 1
                    WHERE prediction_id = ?
                """
                await self.db_manager.execute_query(
                    update_query, (is_winner, payout, prediction_id)
                )

                if is_winner:
                    bt.logging.info(
                        f"Prediction {prediction_id}: Miner {miner_uid} won. Payout: {payout}"
                    )
                else:
                    bt.logging.info(
                        f"Prediction {prediction_id}: Miner {miner_uid} lost."
                    )

            # Ensure entropy scores are calculated
            await self.ensure_predictions_have_entropy_score(external_ids)

        except Exception as e:
            
            bt.logging.error(f"Error in update_predictions_with_payouts: {e}")
            raise

    async def ensure_predictions_have_entropy_score(self, external_ids):
        """Ensure all predictions for given games have entropy scores calculated."""
        try:
            query = """
                SELECT p.prediction_id, p.miner_uid, p.game_id, p.predicted_outcome, 
                       p.predicted_odds, p.wager, p.prediction_date
                FROM predictions p
                WHERE p.game_id IN ({seq})
            """.format(seq=",".join(["?"] * len(external_ids)))
            
            predictions = await self.db_manager.fetch_all(query, tuple(external_ids))
            bt.logging.info(f"Processing {len(predictions)} predictions for entropy scores")
            
            for pred in predictions:
                try:
                    # Add prediction to entropy system
                    self.entropy_system.add_prediction(
                        prediction_id=pred['prediction_id'],
                        miner_uid=pred['miner_uid'],
                        game_id=pred['game_id'],
                        predicted_outcome=pred['predicted_outcome'],
                        wager=float(pred['wager']),
                        predicted_odds=float(pred['predicted_odds']),
                        prediction_date=pred['prediction_date']
                    )
                    bt.logging.debug(f"Added prediction {pred['prediction_id']} to entropy system")
                    
                except Exception as e:
                    bt.logging.error(f"Error adding prediction {pred['prediction_id']} to entropy system: {e}")
                    bt.logging.error(f"Traceback:\n{traceback.format_exc()}")
                    continue
                    
        except Exception as e:
            bt.logging.error(f"Error in ensure_predictions_have_entropy_score: {e}")
            bt.logging.error(f"Traceback:\n{traceback.format_exc()}")

    async def _fix_null_game_data(self):
        """Check for and fix any games with null create_date or game_id values."""
        try:
            bt.logging.info("Starting null game data fix operation")
            
            # Find games with null create_date or game_id
            query = """
                SELECT external_id, game_id, create_date, event_start_date 
                FROM game_data
                WHERE create_date IS NULL OR game_id IS NULL
            """
            
            null_games = await self.db_manager.fetch_all(query, {})
            if not null_games:
                bt.logging.info("No games found with null critical fields.")
                return
                
            bt.logging.info(f"Found {len(null_games)} games with null critical fields. Starting fix...")
            
            # Fix games using a transaction
            try:
                async with self.db_manager.transaction() as session:
                    bt.logging.debug("Transaction started for null game data fix")
                    # Fix each game
                    current_time = datetime.now(timezone.utc).isoformat()
                    for game in null_games:
                        external_id = game.get('external_id')
                        if not external_id:
                            continue
                            
                        # Generate values for null fields
                        game_id = game.get('game_id')
                        if not game_id:
                            game_id = str(uuid.uuid4())
                        
                        # Calculate create_date as event_start_date minus 1 week if null
                        create_date = game.get('create_date')
                        if not create_date:
                            event_start_date = game.get('event_start_date')
                            if event_start_date:
                                try:
                                    # Parse the event start date and subtract 1 week
                                    event_dt = datetime.fromisoformat(event_start_date.replace('Z', '+00:00'))
                                    create_dt = event_dt - timedelta(days=7)
                                    create_date = create_dt.isoformat()
                                except (ValueError, AttributeError):
                                    # Fallback to current time if parsing fails
                                    create_date = current_time
                            else:
                                create_date = current_time
                        
                        # Update the record
                        update_query = """
                            UPDATE game_data
                            SET game_id = :game_id, create_date = :create_date
                            WHERE external_id = :external_id
                        """
                        
                        await session.execute(
                            text(update_query), 
                            {
                                "game_id": game_id,
                                "create_date": create_date,
                                "external_id": external_id
                            }
                        )
                        
                        bt.logging.debug(f"Fixed game with external_id: {external_id}")
                    
                    bt.logging.debug("Transaction completed successfully for null game data fix")
                    
            except asyncio.CancelledError as e:
                bt.logging.error("Transaction cancelled during null game data fix")
                bt.logging.error(f"Cancellation context: {e.__context__}")
                bt.logging.error(f"Cancellation traceback: {traceback.format_exc()}")
                raise
                
            bt.logging.info(f"Successfully fixed {len(null_games)} games with null critical fields.")
            
        except Exception as e:
            bt.logging.error(f"Error fixing null game data: {str(e)}")
            bt.logging.error(traceback.format_exc())

    async def _fix_null_miner_keys(self):
        """Fix null hotkey/coldkey values in miner_stats by fetching current network state."""
        try:
            bt.logging.info("Starting null miner keys fix operation")
            
            # Find miners with null keys
            query = """
                SELECT miner_uid, miner_hotkey, miner_coldkey
                FROM miner_stats
                WHERE miner_hotkey IS NULL OR miner_coldkey IS NULL
                AND miner_uid < 256
            """
            
            null_miners = await self.db_manager.fetch_all(query, {})
            if not null_miners:
                bt.logging.info("No miners found with null hotkey/coldkey values.")
                return
                
            bt.logging.info(f"Found {len(null_miners)} miners with null hotkey/coldkey values. Starting fix...")
            
            # Get current network state
            try:
                bt.logging.debug("Fetching current network state from subtensor")
                # Get current network state from subtensor
                subtensor = bt.subtensor()
                neurons = subtensor.neurons(netuid=self.netuid)
                
                # Create mapping of UID to hotkey/coldkey
                uid_to_keys = {}
                for neuron in neurons:
                    if hasattr(neuron, 'uid') and hasattr(neuron, 'hotkey') and hasattr(neuron, 'coldkey'):
                        uid_to_keys[neuron.uid] = {
                            'hotkey': neuron.hotkey,
                            'coldkey': neuron.coldkey
                        }
                
                bt.logging.info(f"Retrieved {len(uid_to_keys)} current network mappings")
                
                # Fix miners using a transaction
                try:
                    async with self.db_manager.transaction() as session:
                        bt.logging.debug("Transaction started for null miner keys fix")
                        for miner in null_miners:
                            miner_uid = miner.get('miner_uid')
                            if miner_uid not in uid_to_keys:
                                bt.logging.warning(f"No current network mapping found for miner {miner_uid}")
                                continue
                                
                            keys = uid_to_keys[miner_uid]
                            
                            # Update the record
                            update_query = """
                                UPDATE miner_stats
                                SET miner_hotkey = :hotkey,
                                    miner_coldkey = :coldkey
                                WHERE miner_uid = :miner_uid
                            """
                            
                            await session.execute(
                                text(update_query), 
                                {
                                    "hotkey": keys['hotkey'],
                                    "coldkey": keys['coldkey'],
                                    "miner_uid": miner_uid
                                }
                            )
                            
                            bt.logging.debug(f"Fixed keys for miner {miner_uid}")
                        
                        bt.logging.debug("Transaction completed successfully for null miner keys fix")
                        
                except asyncio.CancelledError as e:
                    bt.logging.error("Transaction cancelled during null miner keys fix")
                    bt.logging.error(f"Cancellation context: {e.__context__}")
                    bt.logging.error(f"Cancellation traceback: {traceback.format_exc()}")
                    raise
                
                bt.logging.info(f"Successfully fixed {len(null_miners)} miners with null hotkey/coldkey values.")
                
            except Exception as e:
                bt.logging.error(f"Error fetching network state: {str(e)}")
                bt.logging.error(traceback.format_exc())
            
        except Exception as e:
            bt.logging.error(f"Error fixing null miner keys: {str(e)}")
            bt.logging.error(traceback.format_exc())



