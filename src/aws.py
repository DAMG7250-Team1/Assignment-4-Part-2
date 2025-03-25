import boto3
import logging

# Set up logging
logger = logging.getLogger(__name__)

class s3:
    @staticmethod
    def get_s3_client():
        """
        Get a boto3 S3 client using default credentials
        """
        try:
            return boto3.client('s3')
        except Exception as e:
            logger.error(f"Error creating S3 client: {e}")
            raise 