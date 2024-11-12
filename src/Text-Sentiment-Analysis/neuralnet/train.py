import comet_ml
import os 
import torch
import dataset, engine
import model
import argparse
from comet_ml import Experiment

# Load API
from dotenv import load_dotenv
load_dotenv()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the dataset
    df = dataset.load_data()
    df = dataset.clean(df)
    X_train, y_train, X_test, y_test = dataset.split_data(df, test_size=0.2)

    # Create dataloaders
    train_loader, test_loader, class_names = dataset.create_dataloaders(X_train,
                                                                        y_train,
                                                                        X_test,
                                                                        y_test, 
                                                                        args.batch_size, 
                                                                        args.num_workers) 

    # Create model
    if args.model.lower() == "lstm":
        sentiment_model = model.LSTMModel(input_dim=300,
                                          hidden_dim=args.hidden_units, 
                                          output_dim=len(class_names),
                                          num_layers=args.num_layers, 
                                          bidirectional=args.bidirectional,
                                          dropout=args.dropout) 
    elif args.model.lower() == "gru":
        sentiment_model = model.GRUModel(input_dim=300,
                                         hidden_dim=args.hidden_units, 
                                         output_dim=len(class_names),
                                         num_layers=args.num_layers, 
                                         bidirectional=args.bidirectional,
                                         dropout=args.dropout) 

    # Setup loss_fn and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sentiment_model.parameters(),
                                 lr=args.learning_rate) 

    # Start the training through engine.py
    print("Training...")
    engine.train(model=sentiment_model,
                 train_dataloader=train_loader,
                 test_dataloader=test_loader,
                 classes=class_names,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 clip=5.0,
                 epochs=args.num_epochs, 
                 device=device,
                 experiment=experiment)


if __name__ == "__main__":
    # Initialize Comet ML Experiment
    comet_ml.login(api_key=os.getenv('COMET_API_KEY'),
                   project_name="tweet-sentiment-analysis")

    experiment = comet_ml.Experiment()

    # Setup argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Tweet Sentiment Analysis Training")

    # Model parameter
    parser.add_argument("--model", type=str, default="gru", help="Model selection (default: GRU)") 
    parser.add_argument("--bidirectional", "-bi", type=bool, default=False, help="Bidirectional status (default: True)") 
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--hidden_units", type=int, default=128, help="Number of hidden units in LSTM (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers (default: 2)")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--num_epochs", "-epochs", type=int, default=25, help="Number of epochs (default: 25)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker threads for data loading (default: 2)")

    # Parse the arguments
    args = parser.parse_args()
    main(args)

    # Log the model to Comet ML Experiment
    experiment.log_model(
        name="Tweet_Sentiment_Analysis_Model",
        file_or_folder="best_model.pth"
    )

    # End the experiment
    experiment.end()
