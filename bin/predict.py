import argparse

from cyclegan_keras.cyclegan import PredictionModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main = parser.add_argument_group('main')
    main.add_argument('--data-dir', required=True, help='path to images')
    main.add_argument('--output-dir', required=True, help='path to save result images to')
    main.add_argument('--output-suffix', default='', help='suffix to add to result images')
    main.add_argument('--model-dir', type=str, required=True,
                      help=('Directory where model definition files are saved during training. Models will be '
                            'loaded from `model_dir`/`experiment_name`_{G_A,G_B,D_A,D_B}_epoch##.h5'))
    main.add_argument('--experiment-name', type=str, required=True,
                      help=('Experiment name for use in naming model definition files. Models will be loaded '
                            'from `model_dir`/`experiment_name`_{G_A,G_B,D_A,D_B}_epoch##.h5'))
    main.add_argument('--which-epoch', type=str, default='latest',
                      help='Which epoch to load? Set to "latest" to use latest saved model')
    main.add_argument('--which-direction', type=str, default='AtoB', help='AtoB or BtoA')
    main.add_argument('--batch-size', type=int, default=32, help='number of images to predict at once')
    
    args = parser.parse_args()

    predict_model = PredictionModel(args.model_dir, args.experiment_name, args.which_epoch, args.which_direction)
    predict_model.predict(args.data_dir, args.output_dir, args.output_suffix, args.batch_size)
