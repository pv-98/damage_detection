

## Dataset Directory Structure
The directory structure of the dataset should look like,
```
Root
│   
│      
│
└──Vehicle_Damage_Detection_Dataset
   │   
   │   
   │
   └───annotations
   │   
   │   
   └───images
   

```
## Use of the Repository
1. Install the dependencies after cloning the repo by running the command `pip install -r requirements.txt`.
2. To perform the training, run the command, `python main.py --train`.
3. To perform the testing, run the command, `python main.py --evaluate`.
4. The endpoint for the model inference has been defined inside the inference directory.
5. To run the model endpoint locally run, `python main.py --serve`.

The outputs are saved in the output directory along with visualizations

### To test the API created for the model endpoint

#### Description
This endpoint accepts an image file and returns predictions for objects detected in the image. The predictions include bounding boxes, labels, and confidence scores for each detected object.

#### Request
- [**URL**: /predict/](https://damage-detection-447027078795.us-central1.run.app/predict/)
- **Method**: POST
- **Content-Type**: multipart/form-data

file

: The image file to be analyzed. This should be provided as a form-data file upload.

#### Example Request
```bash
curl -X POST "https://damage-detection-447027078795.us-central1.run.app/predict/" -F "file=@path_to_your_image.jpg"
```

Can also use Postman to give request

#### Response
- **Content-Type**: application/json
- **Body**:
  - predictions: A list of prediction objects, each containing:
    - 

box

: A list of four float values representing the coordinates of the bounding box [x_min, y_min, x_max, y_max].
    - 

label

: The label of the detected object.
    - 
score
: The confidence score of the prediction.

#### Example Response
```json
{
  "predictions": [
    {
      "box": [34.5, 23.4, 123.4, 234.5],
      "label": "minor-scratch",
      "score": 0.95
    }
    
  ]
}
```

#### Notes
- The endpoint uses a pre-trained model to perform object detection on the provided image.
- Only predictions with a confidence score greater than 0.5 are included in the response.

