

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

