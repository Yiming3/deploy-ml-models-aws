**Readme drafted with gpt-4o**

# ML Model Deployment on AWS

This project is an educational initiative aimed at deploying machine learning models detecting fake news on AWS. It demonstrates the process of building, saving, and deploying a simple text classification model using Flask and Scikit-learn.

## Project Structure

- **application.py**: The main Flask application that serves the model and handles prediction requests.
- **basic_classifier.pkl**: A pre-trained Naive Bayes classifier model.
- **count_vectorizer.pkl**: A pre-trained CountVectorizer for text feature extraction.

## Prerequisites

- Python 3.x
- Flask
- Scikit-learn
- AWS account for deployment

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application locally**:
   ```bash
   python application.py
   ```

   The application will be available at `http://localhost:5000`.

## Usage

- Send a POST request to `/predict` with a JSON payload containing the text to classify:
  ```json
  {
    "text": "Your text here"
  }
  ```

- The application will return a JSON response with the prediction.


## License

This project is licensed under the MIT License 

## Acknowledgments

- Flask for providing a simple and powerful web framework.
- Scikit-learn for the machine learning tools.
- AWS for cloud infrastructure.

