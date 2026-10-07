"""Check setup and run a deliberately naive baseline; no ML solution is supplied."""
from pathlib import Path
import argparse
import numpy as np
from rover_starter import load_bundle, split_data, evaluate_practice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_dir', nargs='?', type=Path,
                        default=Path(__file__).parent / 'demo_data')
    args = parser.parse_args()
    dev, calibration, mission, config = load_bundle(args.bundle_dir)
    training, validation, fit_rows, train_rows, validation_rows = split_data(dev, calibration)
    sensors = config['sensor_columns']
    print(f'Development: {len(dev)} rows; mission: {len(mission)} rows')
    print(f'Fit rows: {len(fit_rows)}; labeled training: {len(training)}; validation: {len(validation)}')
    print('Sensors:', ', '.join(sensors))
    print(dev[sensors].head().to_string(index=False))
    # Setup demonstration only: replace these constant labels with model predictions.
    predictions = np.full(len(dev), 'firm')
    print('\nNaive baseline (every practice cell predicted firm):')
    print(evaluate_practice(predictions, args.bundle_dir))
    print('\nNow create your own script or notebook and build your learning methods.')


if __name__ == '__main__':
    main()
