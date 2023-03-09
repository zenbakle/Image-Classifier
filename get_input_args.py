import argparse
def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='path to folder of images')
    parser.add_argument('--save_dir', default='save checkpoints/', type=str, help="Set directory to save checkpoints")
    parser.add_argument('--arch', default="vgg", type=str, help="Choose architecture")
    parser.add_argument('--learning_rate', default=0.01, type=int, help="Set hyperparameter learning_rate")
    parser.add_argument('--hidden_units', default=512, type=int, help="Set hyperparameters: hidden_units")
    parser.add_argument('--epochs', default=9, type=int, help="Set hyperparameters: epochs")
    parser.add_argument('--gpu', default="cpu", type=str, help="Use GPU for training")
    return parser.parse_args()


def predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', type=str, help='path to images to predict')
    parser.add_argument('checkpoint', type=str, help="name of checkpoint")
    parser.add_argument('--top_k', default=5, type=int, help="Return top K most likely classes")
    parser.add_argument('--category_names', default="cat_to_name.json", type=str,
                        help="Use a mapping of categories to real names")
    parser.add_argument('--gpu', default="cpu", type=str, help="Use GPU for test")

    return parser.parse_args()