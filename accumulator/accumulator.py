import json


class Accumulator(object):
    def __init__(self):
        self._registered = {}
        self._results = {}

    def register(self, func):
        """Adds a marked function to the list of functions to watch. Modifies the function
        decorator in order to point it at this accumulator."""
        self._registered[func.__name__] = func

    def compute_metrics(self):
        for func_name in self._registered:
            samples = self._registered[func_name].samples
            if len(samples) == 0:
                raise Exception("No samples have been collected.")

            num_samples = len(samples)
            self._results[func_name] = {
                "mean": sum(samples) / num_samples,
                "num_samples": num_samples,
            }

        return self._results

    def as_json(self, **kwargs):
        results = self.compute_metrics()
        return json.dumps(results, **kwargs)

    def show(self):
        print(self.as_json(sort_keys=False, indent=4))
