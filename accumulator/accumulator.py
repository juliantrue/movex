import json


class Accumulator(object):
    def __init__(self):
        self._watched = {}

    def register(self, *funcs):
        """Adds a marked function to the list of functions to watch. Modifies the function
        decorator in order to point it at this accumulator."""

        for func in funcs:
            # if func.accumulator is None:
            #    print(
            #        f"{func.__name__} has not been decorated with a collection decorator."
            #    )
            #    print("Skipping")

            self._watched[func.__name__] = []
            setattr(func, "accumulator", self)

        return self

    def as_json(self, **kwargs):
        return json.dumps(self._watched, **kwargs)

    def show(self):
        print(self.as_json(sort_keys=False, indent=4))
