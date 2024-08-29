# Real-Estate-Pricing

## Project Organization
Please refer to the `file_structure.txt` for an overview of the expected directory structure.

## Getting Started

### Clone the Repository
```cmd
git clone https://github.com/jaideepmurkute/Real-Estate-Pricing.git
cd Real-Estate-Pricing
```

## Development and Testing
### Without using Docker
#### Backend
```cmd
cd real_estate_api
uvicorn app.main:app --host=0.0.0.0 --port=8000
```
#### Frontend
```
cd real_estate_api
npm run serve
```
##### Access the application at: http://localhost:8080/

### Using Docker
#### Backend
```
cd real_estate_api

# Create the backend's Docker image
docker-compose build    

# Start the backend server
docker-compose up    

# After done using; bring down services and remove images
docker-compose down --rmi all    
```
#### Frontend
```
cd real_estate_frontend

# Create the frontend's Docker image
docker-compose build    

# Start the frontend server
docker-compose up    

# Access the application at: http://localhost:8080/

# After done using; bring down services and remove images
docker-compose down --rmi all    
```

## Tech Stack
* Database
  * MySQL, SQLAlchemy
* Backend
  * Core Programming: Python
  * Machine/Deep Learning: TensorFlow
  * REST Data API: FastAPI
* Frontend: 
  * VueJS, HTML, CSS
* Testing: 
  * PyTest
* Version Control: 
  * Git
* Deployment: 
  * Docker, Uvicorn, npm
* Documentation: 
  * WIP
* Other Important Libraries:
  * Data handling & Numerical functions: Pandas, Nump


## License


