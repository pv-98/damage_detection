


from training_pipeline.train import main as train_main
from testing.test import main as eval_main
#from inference.model_endpoint import app  

import uvicorn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Damage Detection Project")
    parser.add_argument('--train', action='store_true', help="Run training pipeline")
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation pipeline")
    parser.add_argument('--serve', action='store_true', help="Run the FastAPI model endpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train_main()
    if args.evaluate:
        eval_main()
    if args.serve:
        from inference.model_endpoint import app
        uvicorn.run(app, host='0.0.0.0', port=8000)
    if not (args.train or args.evaluate or args.serve):
        print("Please specify --train and/or --evaluate and/or --serve")

