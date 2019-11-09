from src.models.mapf_model import MapfModel
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from src.metrics import coverage_score, cumsum_score


class Baselines(MapfModel):

    def print_results(self, results_file='model-results.csv', notes=''):
        with open(results_file, 'a+', newline='') as csvfile:
            fieldnames = ['Model', 'Accuracy', 'Coverage', 'Cumsum(minutes)', 'Notes']
            res_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            res_writer.writeheader()
            random_preds = [self.conversions[x] for x in np.random.randint(0, 6, size=(len(self.X_test)))]
            random_acc = accuracy_score(self.y_test, random_preds)
            random_coverage = coverage_score(self.X_test, random_preds, self.max_runtime)
            random_cumsum = cumsum_score(self.X_test, random_preds)
            res_writer.writerow({'Model': "Random baseline",
                                 'Accuracy': "{0:.2%}".format(random_acc),
                                 'Coverage': "{0:.2%}".format(random_coverage),
                                 'Cumsum(minutes)': int(random_cumsum),
                                 'Notes': notes})

            for key, conversion in self.conversions.items():
                preds = [conversion] * len(self.X_test)
                alg_acc = accuracy_score(self.y_test, preds)
                alg_coverage = coverage_score(self.X_test, preds)
                alg_cumsum = cumsum_score(self.X_test, preds)
                res_writer.writerow({'Model': conversion,
                                     'Accuracy': "{0:.2%}".format(alg_acc),
                                     'Coverage': "{0:.2%}".format(alg_coverage),
                                     'Cumsum(minutes)': int(alg_cumsum),
                                     'Notes': notes})

            optimal_acc = accuracy_score(self.y_test, self.X_test['Y'])
            optimal_coverage = coverage_score(self.X_test, self.X_test['Y'])
            optimal_cumsum = cumsum_score(self.X_test, self.X_test['Y'])

            res_writer.writerow({'Model': 'Optimal Oracle',
                                 'Accuracy': "{0:.2%}".format(optimal_acc),
                                 'Coverage': "{0:.2%}".format(optimal_coverage),
                                 'Cumsum(minutes)': int(optimal_cumsum),
                                 'Notes': notes})
