"""
R Service Wrapper - Provides seamless integration with R service via pyRserve
Replaces rpy2 functionality for microservices architecture
"""
# Suppress deprecation warnings from pkg_resources
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import io
import os
import queue
import threading
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyRserve

# R service configuration
R_SERVICE_URL = os.getenv('R_SERVICE_URL', 'http://metamap-r:6311')
R_HOST = R_SERVICE_URL.split('://')[-1].split(':')[0]
R_PORT = int(R_SERVICE_URL.split(':')[-1])
R_POOL_SIZE = int(os.getenv('R_POOL_SIZE', '1'))  # Single connection (reduces startup time)

class RConnection:
    """Manages connection to R service"""
    
    def __init__(self):
        self.conn = None
        self.connect()
        
    def connect(self):
        """Establish connection to R service"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempting to connect to R service at {R_HOST}:{R_PORT} (attempt {attempt + 1}/{max_retries})")
                self.conn = pyRserve.connect(host=R_HOST, port=R_PORT)
                # Source the R support file
                self.conn.r.source('/app/src/utils/r_support.R')
                print(f"✅ Connected to R service at {R_HOST}:{R_PORT}")
                return
            except Exception as e:
                print(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to connect to R service after {max_retries} attempts")
                    raise
            
    def ensure_connected(self):
        """Ensure connection is alive, reconnect if needed"""
        try:
            # Simple test to check if connection is alive
            self.conn.eval('1+1')
        except:
            print("Reconnecting to R service...")
            self.connect()
    
    def call_function(self, func_name, *args, **kwargs):
        """
        Call an R function with arguments
        
        Args:
            func_name: Name of the R function
            *args: Positional arguments (DataFrames will be converted)
            **kwargs: Keyword arguments
        """
        self.ensure_connected()
        
        # Convert arguments and build R code
        temp_vars = []
        converted_args = []
        
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                # Assign DataFrame to a temporary R variable using textConnection
                temp_var = f"pydf_{i}"
                temp_vars.append(temp_var)
                
                # Convert DataFrame to CSV string
                csv_buffer = io.StringIO()
                arg.to_csv(csv_buffer, index=False)
                csv_lines = csv_buffer.getvalue().split('\n')
                
                # Build R vector of lines
                csv_lines_r = [line.replace('"', '\\"') for line in csv_lines if line]
                csv_vector = ', '.join([f'"{line}"' for line in csv_lines_r])
                
                # Use textConnection and read.csv
                assign_code = f'{temp_var} <- read.csv(textConnection(c({csv_vector})), stringsAsFactors=FALSE, check.names=FALSE)'
                
                # Execute assignment in R
                self.conn.voidEval(assign_code)
                converted_args.append(temp_var)
                
            elif isinstance(arg, str):
                # Quote strings properly
                converted_args.append(f"'{arg}'")
            elif isinstance(arg, (int, float, bool)):
                converted_args.append(str(arg))
            elif isinstance(arg, (list, np.ndarray)):
                # Convert to R vector
                temp_var = f"pyvec_{i}"
                temp_vars.append(temp_var)
                vec_vals = list(arg) if isinstance(arg, np.ndarray) else arg
                vec_code = f"{temp_var} <- c({','.join(map(str, vec_vals))})"
                self.conn.voidEval(vec_code)
                converted_args.append(temp_var)
            else:
                converted_args.append(str(arg))
        
        # Build R function call
        args_str = ', '.join(converted_args)
        r_code = f"{func_name}({args_str})"
        
        print(f"🔧 Calling R function: {r_code[:100]}...")
        
        try:
            # Execute R code
            result = self.conn.eval(r_code)
            
            # Debug output
            print(f"   ✓ Result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') and not isinstance(result, str) else 'N/A'}")
            
            # Convert result back to Python
            py_result = self.r2py(result)
            print(f"   ✓ Converted to Python: type={type(py_result)}, shape={py_result.shape if hasattr(py_result, 'shape') else 'N/A'}")
            
            return py_result
        except Exception as e:
            # Better error handling with context
            error_msg = str(e)
            raise RuntimeError(
                f"R function '{func_name}' failed: {error_msg}\n"
                f"R code: {r_code[:200]}..."
            ) from e
        finally:
            # Clean up temporary variables with better error handling
            for temp_var in temp_vars:
                try:
                    self.conn.voidEval(f"rm({temp_var})")
                except Exception as cleanup_err:
                    print(f"⚠️  Warning: Failed to cleanup temp var {temp_var}: {cleanup_err}")
    
    def r2py(self, r_obj):
        """Convert R object to Python object"""
        # Handle different R object types
        if r_obj is None:
            return None
        
        # pyRserve returns different types:
        # - TaggedList for data.frames and lists
        # - TaggedArray for named vectors  
        # - Array for unnamed vectors
        # - scalar types (int, float, str, bool)
        
        try:
            # Check if it's a dict-like (TaggedList for data.frames or named lists)
            if isinstance(r_obj, dict) or hasattr(r_obj, 'keys'):
                # Try to detect if it's a list vs dataframe
                result_dict = dict(r_obj)
                
                # If all values are 1D arrays/lists of same length, it's likely a dataframe
                if result_dict:
                    try:
                        return pd.DataFrame(result_dict)
                    except:
                        # If DataFrame fails, return as dict with converted values
                        return {k: self.r2py(v) for k, v in result_dict.items()}
            
            # Check if it's a pyRserve Array or list-like
            if hasattr(r_obj, '__iter__') and not isinstance(r_obj, (str, dict)):
                # Convert to numpy array, preserving numeric types
                arr = np.array(list(r_obj))
                # If it's a single-element array, return scalar
                if arr.shape == (1,):
                    return arr[0]
                return arr
            
            # Return as-is for scalars
            return r_obj
        except Exception as e:
            print(f"⚠️  Warning: r2py conversion failed: {e}, returning as-is")
            return r_obj
    
    def eval(self, code):
        """Evaluate R code"""
        self.ensure_connected()
        return self.conn.eval(code)
    
    def eval_void(self, code):
        """Evaluate R code without waiting for return value (for long-running operations)"""
        self.ensure_connected()
        self.conn.voidEval(code)
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()

# Connection pool for thread-safe concurrent access
class RConnectionPool:
    """Thread-safe connection pool for R service"""
    
    def __init__(self, size=R_POOL_SIZE):
        self.pool = queue.Queue(maxsize=size)
        self.size = size
        self._lock = threading.Lock()
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of connection pool"""
        with self._lock:
            if not self._initialized:
                print(f"🔧 Initializing R connection pool with {self.size} connections...")
                for i in range(self.size):
                    try:
                        conn = RConnection()
                        self.pool.put(conn)
                        print(f"   ✓ Connection {i+1}/{self.size} established")
                    except Exception as e:
                        print(f"   ✗ Failed to create connection {i+1}: {e}")
                        # Add at least one connection or fail
                        if i == 0:
                            raise
                self._initialized = True
                print(f"✅ Connection pool ready with {self.pool.qsize()} connections")
    
    @contextmanager
    def get_connection(self, timeout=10):
        """Get a connection from the pool (context manager)"""
        if not self._initialized:
            self._initialize()
        
        conn = None
        try:
            conn = self.pool.get(timeout=timeout)
            yield conn
        except queue.Empty:
            raise TimeoutError(f"Could not acquire R connection within {timeout}s")
        finally:
            if conn is not None:
                self.pool.put(conn)
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

# Global connection pool
_r_pool = None

def get_r_connection_pool():
    """Get or create global connection pool"""
    global _r_pool
    if _r_pool is None:
        _r_pool = RConnectionPool()
    return _r_pool

def call_r_function(func_name, *args, **kwargs):
    """
    Convenience function to call R function (thread-safe)
    
    Usage:
        result = call_r_function('cut_var', dataframe, cutoff_value)
    """
    pool = get_r_connection_pool()
    with pool.get_connection() as conn:
        return conn.call_function(func_name, *args, **kwargs)

def source_r_file(filepath):
    """Source an R file (thread-safe)"""
    pool = get_r_connection_pool()
    with pool.get_connection() as conn:
        conn.eval(f"source('{filepath}')")

# Compatibility layer for rpy2-style code
class GlobalEnv:
    """Mimic rpy2.robjects.globalenv"""
    def __getitem__(self, func_name):
        """Get R function by name"""
        def wrapper(*args, **kwargs):
            return call_r_function(func_name, *args, **kwargs)
        return wrapper

# Create instance to use
globalenv = GlobalEnv()

# Pandas conversion utilities (rpy2-compatible interface)
class pandas2ri:
    """Pandas to R conversion utilities"""
    
    @staticmethod
    def activate():
        """Compatibility method - does nothing in this implementation"""
        pass
    
    @staticmethod
    def py2rpy(df):
        """Convert pandas DataFrame to R - returns the DataFrame for passing to functions"""
        return df
    
    @staticmethod
    def rpy2py(r_obj):
        """Convert R object to Python"""
        conn = get_r_connection()
        return conn.r2py(r_obj)


def initialize_pool():
    """Pre-initialize the R connection pool at startup"""
    try:
        pool = get_r_connection_pool()
        pool._initialize()
    except Exception as e:
        print(f"⚠️  R connection pool initialization deferred: {e}")
