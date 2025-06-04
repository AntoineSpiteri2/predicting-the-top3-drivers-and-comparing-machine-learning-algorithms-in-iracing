import irsdk
import time

class IRacingSDK:
    """
    A wrapper class for handling iRacing SDK connection.
    """
    def __init__(self):
        """
        Initializes the IRacingSDK object and establishes the connection.
        """
        self.ir = irsdk.IRSDK()
        self.connect()


    def connect(self):
        """
        Starts up the iRacing SDK connection and checks if the connection is successful.
        """
        try:
            self.ir.startup()
            if not self.ir.is_connected:
                raise ConnectionError("Failed to connect to iRacing SDK.")
        except Exception as e:
            print(f"Error during SDK connection: {e}")
            raise

    def get_connection(self):
        """
        Returns the iRacing SDK connection object.
        
        Returns:
            irsdk.IRSDK: The active iRacing SDK object.
        """
        return self.ir


# Example usage:
def get_iracing_sdk_connection(retries=3, delay=5):
    """
    Returns an instance of IRacingSDK for further use, retrying if the connection fails.
    
    Args:
        retries (int): Number of times to retry the connection.
        delay (int): Delay in seconds between retries.
    
    Returns:
        IRacingSDK: An instance of the iRacing SDK wrapper, or None if connection fails.
    """

    for attempt in range(retries):
        try:
            sdk = IRacingSDK()
            return sdk.get_connection()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Unable to establish iRacing SDK connection: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    
    raise ConnectionError("All attempts to connect to iRacing SDK have failed.")




