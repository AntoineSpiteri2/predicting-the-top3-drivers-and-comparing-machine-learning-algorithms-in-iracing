from iracingdataapi.client import irDataClient
import asyncio


# Credentials for iRacing API (consider using environment variables for better security)
USERNAME = 'youremail' # Replace with your iRacing username
PASSWORD = 'yourpassword' # Replace with your iRacing password

def return_iracing_api_client():
    """
    Initializes and returns an iRacing Data API client.

    Returns:
        irDataClient: An authenticated iRacing Data API client instance.
    """
    try:
        # Initialize the iRacing Data API client with credentials
        client = irDataClient(username=USERNAME, password=PASSWORD)
        print("iRacing API client initialized successfully.")
        return client
    except Exception as e:
        # Catch and display any errors during the client initialization
        print(f"Error initializing iRacing API client: {e}")
        return None



import asyncio

class AsyncIDClientWrapper:
    def __init__(self, idc_instance, max_concurrent_requests=4):
        """
        Wraps an irDataClient instance with asynchronous support and rate limiting.
        
        Args:
            idc_instance: The irDataClient instance to wrap.
            max_concurrent_requests: Maximum concurrent API requests allowed.
        """
        self._idc = idc_instance
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    def __getattr__(self, name):
        """
        Intercept attribute access and return an asynchronous wrapper 
        for callable attributes (methods) with rate limiting.
        """
        attr = getattr(self._idc, name)
        if callable(attr):
            async def async_method(*args, **kwargs):
                async with self._semaphore:  # Enforce rate limit
                    return await asyncio.to_thread(attr, *args, **kwargs)
            return async_method
        return attr
