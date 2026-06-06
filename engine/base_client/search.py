import functools
import random
import itertools
import time
from multiprocessing import Process, Queue
from typing import Iterable, List, Optional, Tuple
from itertools import islice

import numpy as np
import tqdm
import os
from ml_dtypes import bfloat16

from dataset_reader.base_reader import Query
from engine.base_client.utils import check_data_type

DEFAULT_TOP = 10
MAX_QUERIES = int(os.getenv("MAX_QUERIES", -1))



class BaseSearcher:
    _doc_id_counter = None  # Will be initialized per process
    MP_CONTEXT = None

    def __init__(self, host, connection_params, search_params):
        self.host = host
        self.connection_params = connection_params
        self.search_params = search_params

    @classmethod
    def init_client(
        cls, host: str, distance, connection_params: dict, search_params: dict
    ):
        raise NotImplementedError()

    @classmethod
    def get_mp_start_method(cls):
        return None

    @classmethod
    def search_one(
        cls, vector: List[float], meta_conditions, top: Optional[int]
    ) -> List[Tuple[int, float]]:
        raise NotImplementedError()

    @classmethod
    def insert_one(cls, doc_id: str, vector: List[float], meta_conditions):
        raise NotImplementedError()

    @classmethod
    def update_one(cls, doc_id: str, vector: List[float], meta_conditions):
        raise NotImplementedError()

    @classmethod
    def wait_for_index_sync(cls, verbose=True):
        """
        Wait for inserted documents to be fully indexed.
        Default implementation does nothing. Override in engine-specific clients.
        """
        pass

    @classmethod
    def _search_one(cls, query, top: Optional[int] = None):
        if top is None:
            top = (
                len(query.expected_result)
                if query.expected_result is not None and len(query.expected_result) > 0
                else DEFAULT_TOP
            )

        start = time.perf_counter()
        search_res = cls.search_one(query.vector, query.meta_conditions, top)
        end = time.perf_counter()

        precision = 1.0
        if query.expected_result:
            ids = set(x[0] for x in search_res)
            precision = len(ids.intersection(query.expected_result[:top])) / top
        return precision, end - start

    @classmethod
    def _get_doc_id_counter(cls):
        if cls._doc_id_counter is None:
            # Use process ID to create unique starting point for each worker
            process_id = os.getpid()
            # Each process gets a unique range: 1000000000 + (pid * 1000000)
            start_offset = 1000000000 + (process_id % 1000) * 1000000
            cls._doc_id_counter = itertools.count(start_offset)
        return cls._doc_id_counter

    @classmethod
    def _insert_one(cls, query):
        start = time.perf_counter()

        # Generate unique doc_id with process-safe counter
        doc_id = next(cls._get_doc_id_counter())

        cls.insert_one(str(doc_id), query.vector, query.meta_conditions)
        end = time.perf_counter()
        # No precision metric for inserts, so precision=1.0
        return 1.0, end - start

    @classmethod
    def _update_one(cls, query, dataset_size):
        """
        Update an existing document with a new vector.
        
        Picks a random doc_id from [0, dataset_size-1] which corresponds to
        the keys created during the initial upload (upload uses str(idx) as keys).
        
        Args:
            query: Query object containing the vector to use for the update
            dataset_size: Number of documents uploaded during initial load
        """
        start = time.perf_counter()

        # Pick a random doc_id from the existing dataset
        # Keys are created as str(0), str(1), ..., str(dataset_size-1) during upload
        if dataset_size <= 0:
            raise ValueError(
                "dataset_size must be > 0 for updates. "
                "Ensure the dataset config has 'vector_count' set, or check upload_end_idx."
            )
        doc_id = random.randint(0, dataset_size - 1)

        cls.update_one(str(doc_id), query.vector, query.meta_conditions)
        end = time.perf_counter()
        # No precision metric for updates, so precision=1.0
        return 1.0, end - start

    def search_all(
        self,
        distance,
        queries: Iterable[Query],
        num_queries: int = -1,
        modify_fraction: float = 0.0,
        modify_operation: str = "insert",
        dataset_size: int = 0,
    ):
        """
        Execute searches with optional concurrent modifications (inserts or updates).
        
        Args:
            distance: Distance metric for the search
            queries: Iterator of Query objects
            num_queries: Number of queries to run (-1 for all)
            modify_fraction: Fraction of operations that should be modifications (0.0-1.0)
            modify_operation: Type of modification - "insert" or "update"
            dataset_size: Size of the dataset (required for updates to pick valid IDs)
        """
        parallel = self.search_params.get("parallel", 1)
        top = self.search_params.get("top", None)
        single_search_params = self.search_params.get("search_params", None)
        if single_search_params:
            data_type = check_data_type(single_search_params.get("data_type", "FLOAT32").upper())
        else:
            data_type = np.float32  # Default data type if not specified
        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        # Reset the doc_id counter to prevent any initialization during client setup
        self.__class__._doc_id_counter = None

        search_one = functools.partial(self.__class__._search_one, top=top)
        insert_one = functools.partial(self.__class__._insert_one)
        update_one = functools.partial(self.__class__._update_one, dataset_size=dataset_size)
        
        # Select the modify operation function based on modify_operation parameter
        if modify_operation == "update":
            modify_one = update_one
            modify_label = "update"
        else:
            modify_one = insert_one
            modify_label = "insert"

        # Convert queries to a list for potential reuse
        # Also, converts query vectors to bytes beforehand, preparing them for sending to client without affecting search time measurements
        queries_list = []
        for query in queries:
            query.vector = np.array(query.vector).astype(data_type).tobytes()
            queries_list.append(query)
        
        # Handle MAX_QUERIES environment variable
        if MAX_QUERIES > 0:
            queries_list = queries_list[:MAX_QUERIES]
            print(f"Limiting queries to [0:{MAX_QUERIES-1}]")

        # Handle num_queries parameter
        if num_queries > 0:
            # If we need more queries than available, use a cycling generator
            if num_queries > len(queries_list) and len(queries_list) > 0:
                print(f"Requested {num_queries} queries but only {len(queries_list)} are available.")
                print(f"Using a cycling generator to efficiently process queries.")

                # Create a cycling generator function
                def cycling_query_generator(queries, total_count):
                    """Generate queries by cycling through the available ones."""
                    count = 0
                    while count < total_count:
                        for query in queries:
                            if count < total_count:
                                yield query
                                count += 1
                            else:
                                break

                # Use the generator instead of creating a full list
                used_queries = cycling_query_generator(queries_list, num_queries)
                # We need to know the total count for the progress bar
                total_query_count = num_queries
            else:
                used_queries = queries_list[:num_queries]
                total_query_count = len(used_queries)
                print(f"Using {num_queries} queries")
        else:
            used_queries = queries_list
            total_query_count = len(used_queries)

        # Interval reporting setup
        interval_size = 10000  # Report every 10K operations 
        need_interval_reporting = total_query_count >= interval_size
        interval_counter = 0
        overall_start_time = time.perf_counter()
        
        # Calculate total number of intervals for progress tracking
        total_intervals = (total_query_count + interval_size - 1) // interval_size  # Ceiling division
        
        # Initialize progress bar for intervals if needed (only if output is to terminal)
        if need_interval_reporting and os.isatty(1):  # Check if stdout is a terminal
            interval_pbar = tqdm.tqdm(total=total_intervals, desc="Intervals", unit="interval")
        else:
            interval_pbar = None
        
        # Initialize global doc_id offset to ensure uniqueness across intervals
        # Start from a high offset to avoid conflicts with uploaded dataset doc_ids
        # Most datasets have < 100M records, so starting from 100M should be safe
        global_doc_id_offset = 1000000000
        
        # Overall accumulators
        overall_results = []
        overall_modify_count = 0
        overall_search_count = 0
        overall_modify_latencies = []
        overall_search_latencies = []
        
        # Interval statistics for output file
        interval_stats = []
        
        # Convert generator to iterator for interval processing
        query_iterator = iter(used_queries)
        
        # Process queries in intervals of 10K
        while True:
            # Get next interval chunk (up to 10K queries)
            interval_queries = list(islice(query_iterator, interval_size))
            if not interval_queries:
                break  # No more queries
                
            interval_counter += 1
            current_interval_size = len(interval_queries)
            
            if parallel == 1:
                # Single-threaded execution for this interval
                interval_start = time.perf_counter()
                
                # Force reset and set doc_id counter offset for single-threaded execution
                # This ensures we override any previous initialization
                self.__class__._doc_id_counter = itertools.count(global_doc_id_offset)

                # Process queries for this interval
                interval_results = []
                interval_modify_count = 0
                interval_search_count = 0
                interval_modify_latencies = []
                interval_search_latencies = []
                
                for query in interval_queries:
                    if random.random() < modify_fraction:
                        precision, latency = modify_one(query)
                        interval_modify_count += 1
                        interval_modify_latencies.append(latency)
                        interval_results.append((modify_label, precision, latency))
                    else:
                        precision, latency = search_one(query)
                        interval_search_count += 1
                        interval_search_latencies.append(latency)
                        interval_results.append(('search', precision, latency))

                interval_time = time.perf_counter() - interval_start
            else:
                # Parallel execution for this interval
                # Dynamically calculate chunk size based on current interval size
                chunk_size = max(1, current_interval_size // parallel)

                # For interval queries (always a list), use chunked_iterable
                query_chunks = list(chunked_iterable(interval_queries, chunk_size))

                # Create a queue to collect results
                result_queue = Queue()

                # Create worker processes
                processes = []
                for i, chunk in enumerate(query_chunks):
                    # Calculate unique doc_id offset for this worker in this interval
                    worker_doc_id_offset = global_doc_id_offset + (i * 1000000)
                    process = Process(target=worker_function, args=(self, distance, search_one, modify_one, 
                                                                    chunk, result_queue, modify_fraction, modify_label, worker_doc_id_offset))
                    processes.append(process)

                # Start worker processes
                for process in processes:
                    process.start()

                # Collect results from all worker processes
                interval_results = []
                interval_modify_count = 0
                interval_search_count = 0
                interval_modify_latencies = []
                interval_search_latencies = []
                min_start_time = time.perf_counter()

                for _ in processes:
                    proc_start_time, chunk_results, modify_count, search_count, modify_latencies, search_latencies = result_queue.get()
                    interval_results.extend(chunk_results)
                    interval_modify_count += modify_count
                    interval_search_count += search_count
                    interval_modify_latencies.extend(modify_latencies)
                    interval_search_latencies.extend(search_latencies)
                    
                    # Update min_start_time if necessary
                    if proc_start_time < min_start_time:
                        min_start_time = proc_start_time

                # Stop measuring time for the critical work
                interval_time = time.perf_counter() - min_start_time

                # Wait for all worker processes to finish
                for process in processes:
                    process.join()
            
            # Accumulate overall results
            overall_results.extend(interval_results)
            overall_modify_count += interval_modify_count
            overall_search_count += interval_search_count
            overall_modify_latencies.extend(interval_modify_latencies)
            overall_search_latencies.extend(interval_search_latencies)
            
            # Sync modifications to index if there were any in this interval
            if interval_modify_count > 0:
                try:
                    if hasattr(self.__class__, 'sync_inserts'):
                        self.__class__.sync_inserts()
                except Exception as e:
                    print(f"Warning: Failed to sync modifications after interval {interval_counter}: {e}")
            
            # Update global doc_id offset for next interval
            if parallel == 1:
                # For single-threaded, reserve space based on actual modifications in this interval
                global_doc_id_offset += max(1000000, interval_modify_count * 2)  # Some buffer
            else:
                # Reserve space for all parallel workers in this interval
                global_doc_id_offset += parallel * 1000000
            
            # Report interval metrics if needed
            if need_interval_reporting:
                interval_search_precisions = [result[1] for result in interval_results if result[0] == 'search']
                
                # Calculate separate RPS for searches and modifications (inserts or updates)
                search_rps = interval_search_count / interval_time if interval_search_count > 0 else 0
                modify_rps = interval_modify_count / interval_time if interval_modify_count > 0 else 0
                
                # Create interval statistics for output file
                # Use modify_label to name the field appropriately (insert_rps or update_rps)
                interval_stat = {
                    "interval": interval_counter,
                    "operations": current_interval_size,
                    "time_seconds": float(interval_time),  # Ensure it's a float
                    "total_rps": float(current_interval_size / interval_time),  # Overall RPS
                    "search_rps": float(search_rps),  # Search-only RPS
                    "search_min": np.min(interval_search_latencies) if interval_search_latencies else None,
                    "search_p25": np.percentile(interval_search_latencies, 25) if interval_search_latencies else None,
                    "search_p50": np.percentile(interval_search_latencies, 50) if interval_search_latencies else None,
                    "search_p75": np.percentile(interval_search_latencies, 75) if interval_search_latencies else None,
                    "search_p95": np.percentile(interval_search_latencies, 95) if interval_search_latencies else None,
                    "search_p99": np.percentile(interval_search_latencies, 99) if interval_search_latencies else None,
                    "search_max": np.max(interval_search_latencies) if interval_search_latencies else None,
                    f"{modify_label}_rps": float(modify_rps),  # Insert or Update RPS
                    f"{modify_label}_min": np.min(interval_modify_latencies) if interval_modify_latencies else None,
                    f"{modify_label}_p25": np.percentile(interval_modify_latencies, 25) if interval_modify_latencies else None,
                    f"{modify_label}_p50": np.percentile(interval_modify_latencies, 50) if interval_modify_latencies else None,
                    f"{modify_label}_p75": np.percentile(interval_modify_latencies, 75) if interval_modify_latencies else None,
                    f"{modify_label}_p95": np.percentile(interval_modify_latencies, 95) if interval_modify_latencies else None,
                    f"{modify_label}_p99": np.percentile(interval_modify_latencies, 99) if interval_modify_latencies else None,
                    f"{modify_label}_max": np.max(interval_modify_latencies) if interval_modify_latencies else None,
                    "searches": interval_search_count,
                    f"{modify_label}s": interval_modify_count,  # inserts or updates count
                    "search_precision": float(np.mean(interval_search_precisions)) if interval_search_precisions else None
                }
                interval_stats.append(interval_stat)
                
                # Debug: Print number of intervals collected so far
                print(f"DEBUG: Collected {len(interval_stats)} intervals so far", flush=True)
                
                # Update progress bar with separate RPS metrics
                # Use capitalized modify_label for display
                modify_label_cap = modify_label.capitalize()
                if interval_pbar:
                    interval_pbar.update(1)
                    interval_pbar.set_postfix({
                        'Total_RPS': f"{current_interval_size / interval_time:.1f}",
                        'Search_RPS': f"{search_rps:.1f}",
                        f'{modify_label_cap}_RPS': f"{modify_rps:.1f}",
                        'Searches': interval_search_count,
                        f'{modify_label_cap}s': interval_modify_count,
                        'Precision': f"{np.mean(interval_search_precisions):.4f}" if interval_search_precisions else "N/A"
                    })
        
        # Close progress bar when done
        if interval_pbar:
            interval_pbar.close()
            print()  # Add a blank line after progress bar
        
        # Calculate total time for overall metrics
        total_time = time.perf_counter() - overall_start_time
        
        # Use overall accumulated results
        results = overall_results
        total_modify_count = overall_modify_count
        total_search_count = overall_search_count
        all_modify_latencies = overall_modify_latencies
        all_search_latencies = overall_search_latencies

        # Extract overall precisions and latencies
        all_precisions = [result[1] for result in results]
        all_latencies = [result[2] for result in results]

        # Calculate search-only precisions (exclude inserts from precision calculation)
        search_precisions = [result[1] for result in results if result[0] == 'search']

        # Create histogram distributions for latencies with fixed bin ranges for cross-run comparison
        def create_fixed_range_histograms(search_latencies, insert_latencies, num_bins=50):
            """
            Create histograms with fixed bin ranges that are consistent across all runs.
            This allows comparing histograms across different datasets, algorithms, and configurations.
            """
            # Define fixed bin ranges (in seconds) that cover typical latency patterns
            # These ranges are designed to capture:
            # - Fast queries: 0-10ms (common for well-optimized systems)
            # - Medium queries: 10-100ms (typical for complex queries)
            # - Slow queries: 100ms-1s (degraded performance scenarios)
            # - Very slow queries: 1s+ (outliers, timeouts, system issues)
            
            # Use log-spaced bins to capture both fast and slow queries effectively
            # Range: 1ms to 2 seconds (covers typical mixed workload latencies)
            min_latency = 0.001  # 1ms
            max_latency = 2.0    # 2 seconds
            
            # Create logarithmically-spaced bins for better distribution across orders of magnitude
            common_bins = np.logspace(np.log10(min_latency), np.log10(max_latency), num_bins + 1)
            
            # Create histograms using the same fixed bin edges
            search_histogram = None
            if search_latencies:
                hist, _ = np.histogram(search_latencies, bins=common_bins)
                search_histogram = {
                    "counts": hist.tolist(),
                    "bin_edges": common_bins.tolist()
                }
            
            insert_histogram = None
            if insert_latencies:
                hist, _ = np.histogram(insert_latencies, bins=common_bins)
                insert_histogram = {
                    "counts": hist.tolist(),
                    "bin_edges": common_bins.tolist()
                }
            
            return search_histogram, insert_histogram
        
        search_histogram, insert_histogram = create_fixed_range_histograms(
            all_search_latencies if all_search_latencies else None,
            all_modify_latencies if all_modify_latencies else None
        )

        self.__class__.delete_client()


        if len(interval_stats) > 0:
            print(f"DEBUG: First interval: {interval_stats[0]}", flush=True)
            print(f"DEBUG: Last interval: {interval_stats[-1]}", flush=True)

        return {
            # Overall metrics
            "total_time": total_time,
            "total_operations": len(all_latencies),
            "rps": len(all_latencies) / total_time,
            
            # Search metrics
            "search_count": total_search_count,
            "search_rps": total_search_count / total_time if total_search_count > 0 else 0,
            "mean_search_time": np.mean(all_search_latencies) if all_search_latencies else 0,
            "mean_search_precision": np.mean(search_precisions) if search_precisions else 0,
            "p50_search_time": np.percentile(all_search_latencies, 50) if all_search_latencies else 0,
            "p95_search_time": np.percentile(all_search_latencies, 95) if all_search_latencies else 0,
            "p99_search_time": np.percentile(all_search_latencies, 99) if all_search_latencies else 0,
            "search_latency_histogram": search_histogram,
            
            # Insert/Update metrics (labeled as insert for backward compatibility)
            "insert_count": total_modify_count,
            "insert_rps": total_modify_count / total_time if total_modify_count > 0 else 0,
            "mean_insert_time": np.mean(all_modify_latencies) if all_modify_latencies else 0,
            "p50_insert_time": np.percentile(all_modify_latencies, 50) if all_modify_latencies else 0,
            "p95_insert_time": np.percentile(all_modify_latencies, 95) if all_modify_latencies else 0,
            "p99_insert_time": np.percentile(all_modify_latencies, 99) if all_modify_latencies else 0,
            "insert_latency_histogram": insert_histogram,
            
            # Mixed workload metrics
            "actual_modify_fraction": total_modify_count / len(all_latencies) if len(all_latencies) > 0 else 0,
            "target_modify_fraction": modify_fraction,
            "modify_operation": modify_operation,
            
            # Interval statistics (only included if intervals were used)
            "interval_stats": interval_stats if interval_stats else None,
            
            # Legacy compatibility (for existing code that expects these)
            "mean_time": np.mean(all_latencies),
            "mean_precisions": np.mean(search_precisions) if search_precisions else 1.0,  # Only search precisions
            "std_time": np.std(all_latencies),
            "min_time": np.min(all_latencies),
            "max_time": np.max(all_latencies),
            "p50_time": np.percentile(all_latencies, 50),
            "p95_time": np.percentile(all_latencies, 95),
            "p99_time": np.percentile(all_latencies, 99),
            "precisions": search_precisions,  # Only search precisions
            "latencies": all_latencies,
        }

    def setup_search(self):
        pass

    def post_search(self):
        pass

    @classmethod
    def delete_client(cls):
        pass


def chunked_iterable(iterable, size):
    """Yield successive chunks of a given size from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

def process_chunk(chunk, search_one, modify_one, modify_fraction, modify_label):
    results = []
    modify_count = 0
    search_count = 0
    modify_latencies = []
    search_latencies = []
    
    for i, query in enumerate(chunk):
        if random.random() < modify_fraction:
            precision, latency = modify_one(query)
            modify_count += 1
            modify_latencies.append(latency)
            results.append((modify_label, precision, latency))
        else:
            precision, latency = search_one(query)
            search_count += 1
            search_latencies.append(latency)
            results.append(('search', precision, latency))
    
    return results, modify_count, search_count, modify_latencies, search_latencies

# Function to be executed by each worker process
def worker_function(self, distance, search_one, modify_one, chunk, result_queue, modify_fraction=0.0, modify_label="insert", doc_id_offset=0):
    self.init_client(
        self.host,
        distance,
        self.connection_params,
        self.search_params,
    )
    self.setup_search()

    # Force set the doc_id counter offset for this worker (overrides any previous state)
    self.__class__._doc_id_counter = itertools.count(doc_id_offset)

    start_time = time.perf_counter()
    results, modify_count, search_count, modify_latencies, search_latencies = process_chunk(
        chunk, search_one, modify_one, modify_fraction, modify_label
    )
    result_queue.put((start_time, results, modify_count, search_count, modify_latencies, search_latencies))
