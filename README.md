

## Directory Structure
The directory structure of the dataset should look like,
<p>-Root <br>
    -Vehicle_Damage_Detection_Dataset<br>
        -Images<br>
            -Train<br>
                image1.jpg..<br>
            -Val<br>
                image1.jpg..<br>
            -Test<br>
                image1.jpg..<br>
        -Annotations<br>
            instances_train.json<br>
            instances_test.json<br>
            instances_val.json </p>
## Use of the Repository
1. Install the dependencies after cloning the repo by running the command `pip install -r requirements.txt`.
2. To perform the training, run the command, `python main.py --train`.
3. To perform the testing, run the command, `python main.py --evaluate`.
4. The endpoint for the model inference has been defined inside the inference directory.
5. To run the model endpoint locally run, `python main.py --serve`.

The outputs are saved in the output directory along with visualizations

