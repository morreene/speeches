# Speech Database Application

A web application for searching and drafting speeches using AI and embeddings.

## Deployment to Heroku

### Important Note on Data Files

This application relies on data files in the `/data` directory, particularly `speech-text-embedding1.parquet`. 

When deploying to Heroku:

1. For small data files: Make sure they are included in your Git repository.

2. For large files (like parquet files): 
   - If under 500MB: You can use Heroku's persistent storage options or Git LFS
   - If over 500MB: Consider using an external storage service like AWS S3
   
3. The application now includes a fallback mechanism for when data files are not available, so it will start up but with limited functionality.

### Steps for Heroku Deployment

1. Create a Heroku application
   ```
   heroku create
   ```

2. Push your code to Heroku
   ```
   git push heroku main
   ```

3. For data files (optional):
   ```
   heroku plugins:install heroku-builds
   heroku builds:upload --app YOUR_APP_NAME
   ```
   
   Or setup AWS S3 and modify the code to load data from S3.

## Development

Run the application locally:
```
python app.py
```

https://speeches-edbfefc38ad4.herokuapp.com/
