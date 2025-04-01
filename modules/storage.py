"""
Google Drive Storage Module
Handles uploading content and metadata to Google Drive
"""
import os
import logging
import json
from pathlib import Path
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import config

class GoogleDriveStorage:
    """
    Handles uploading video content, images, and metadata to Google Drive
    with proper organization and SEO metadata.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load Google Drive configuration
        self.credentials_file = config.GOOGLE_DRIVE["credentials_file"]
        self.token_file = config.GOOGLE_DRIVE["token_file"]
        self.folder_name = config.GOOGLE_DRIVE["folder_name"]
        
        # Define required scopes
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        
        self.logger.info(f"Initializing GoogleDriveStorage with folder: {self.folder_name}")
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        self.logger.info("Authenticating with Google Drive")
        
        creds = None
        
        # Check if token file exists
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If credentials don't exist or are invalid, refresh them
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    self.logger.error(f"Credentials file not found: {self.credentials_file}")
                    raise FileNotFoundError(f"Google Drive credentials file not found: {self.credentials_file}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save the new credentials
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build the Drive API service
        service = build('drive', 'v3', credentials=creds)
        return service
    
    def _create_folder(self, service, folder_name, parent_id=None):
        """
        Create a folder in Google Drive
        
        Args:
            service: Google Drive API service
            folder_name: Name of the folder to create
            parent_id: Optional parent folder ID
            
        Returns:
            folder_id: ID of the created folder
        """
        self.logger.info(f"Creating folder: {folder_name}")
        
        # Set up folder metadata
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        # Set parent folder if provided
        if parent_id:
            folder_metadata['parents'] = [parent_id]
        
        # Create the folder
        folder = service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        folder_id = folder.get('id')
        self.logger.info(f"Created folder with ID: {folder_id}")
        
        return folder_id
    
    def _get_or_create_folder(self, service, folder_name, parent_id=None):
        """
        Get folder ID if it exists, or create it if not
        
        Args:
            service: Google Drive API service
            folder_name: Name of the folder to find or create
            parent_id: Optional parent folder ID
            
        Returns:
            folder_id: ID of the folder
        """
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = response.get('files', [])
        
        # If folder exists, return its ID
        if files:
            folder_id = files[0].get('id')
            self.logger.info(f"Found existing folder '{folder_name}' with ID: {folder_id}")
            return folder_id
        
        # Otherwise create the folder
        return self._create_folder(service, folder_name, parent_id)
    
    def _upload_file(self, service, file_path, folder_id, metadata=None):
        """
        Upload a file to Google Drive
        
        Args:
            service: Google Drive API service
            file_path: Path to the file to upload
            folder_id: ID of the parent folder
            metadata: Optional metadata for the file
            
        Returns:
            file_id: ID of the uploaded file
            file_url: Direct URL to the file
        """
        file_path = Path(file_path)
        self.logger.info(f"Uploading file: {file_path}")
        
        # Define file metadata
        file_metadata = {
            'name': file_path.name,
            'parents': [folder_id]
        }
        
        # Add any additional metadata
        if metadata:
            # Convert metadata to Drive properties format
            properties = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    properties[key] = str(value)
                elif isinstance(value, list):
                    properties[key] = ','.join(str(item) for item in value)
                else:
                    properties[key] = json.dumps(value)
            
            file_metadata['properties'] = properties
        
        # Determine media type based on file extension
        mimetype = self._get_mimetype(file_path)
        
        # Create media
        media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
        
        # Upload the file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        file_id = file.get('id')
        file_url = file.get('webViewLink')
        
        self.logger.info(f"Uploaded file with ID: {file_id}")
        self.logger.info(f"File URL: {file_url}")
        
        return file_id, file_url
    
    def _get_mimetype(self, file_path):
        """Determine MIME type based on file extension"""
        extension = file_path.suffix.lower()
        
        mimetypes = {
            '.mp4': 'video/mp4',
            '.m4v': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.pdf': 'application/pdf',
            '.md': 'text/markdown'
        }
        
        return mimetypes.get(extension, 'application/octet-stream')
    
    def upload(self, video_file, script, images, metadata, output_dir):
        """
        Upload content to Google Drive
        
        Args:
            video_file: Path to the final video file
            script: Script text or path to script file
            images: List of image paths
            metadata: Video metadata
            output_dir: Directory containing all generated content
            
        Returns:
            dict: Upload results including Drive URLs
        """
        self.logger.info("Starting Google Drive upload process")
        
        try:
            # Authenticate with Google Drive
            service = self._authenticate()
            
            # Get or create main folder
            main_folder_id = self._get_or_create_folder(service, self.folder_name)
            
            # Create a subfolder for this specific video
            video_name = metadata.get('title', 'Untitled Video')
            video_folder_id = self._create_folder(service, video_name, main_folder_id)
            
            # Upload video file
            video_id, video_url = self._upload_file(
                service, 
                video_file, 
                video_folder_id, 
                metadata
            )
            
            # Upload thumbnail if it exists
            thumbnail_url = None
            thumbnail_path = Path(output_dir) / "videos" / "thumbnail.jpg"
            if thumbnail_path.exists():
                thumbnail_id, thumbnail_url = self._upload_file(
                    service,
                    thumbnail_path,
                    video_folder_id
                )
            
            # Upload script if it's a file path
            script_url = None
            if isinstance(script, str) and os.path.isfile(script):
                script_id, script_url = self._upload_file(
                    service,
                    script,
                    video_folder_id
                )
            else:
                # Create a script file and upload it
                script_path = Path(output_dir) / "videos" / "script.txt"
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)
                
                script_id, script_url = self._upload_file(
                    service,
                    script_path,
                    video_folder_id
                )
            
            # Upload metadata JSON
            metadata_path = Path(output_dir) / "videos" / "metadata.json"
            if os.path.isfile(metadata_path):
                metadata_id, metadata_url = self._upload_file(
                    service,
                    metadata_path,
                    video_folder_id
                )
            
            # Create an assets folder and upload images
            assets_folder_id = self._create_folder(service, "Assets", video_folder_id)
            image_urls = []
            
            for image_path in images:
                if os.path.isfile(image_path):
                    _, image_url = self._upload_file(
                        service,
                        image_path,
                        assets_folder_id
                    )
                    image_urls.append(image_url)
            
            # Create a direct link to the Drive folder
            folder_url = f"https://drive.google.com/drive/folders/{video_folder_id}"
            
            self.logger.info(f"Upload complete. Folder URL: {folder_url}")
            
            # Return upload results
            return {
                "drive_url": folder_url,
                "video_url": video_url,
                "thumbnail_url": thumbnail_url,
                "script_url": script_url,
                "image_urls": image_urls,
                "folder_id": video_folder_id
            }
            
        except Exception as e:
            self.logger.error(f"Error uploading to Google Drive: {str(e)}", exc_info=True)
            
            # If upload fails, return information about local files
            return {
                "error": str(e),
                "local_video": video_file,
                "local_dir": str(output_dir)
            }
