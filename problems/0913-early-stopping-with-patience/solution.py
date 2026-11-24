class EarlyStopping:
    def __init__(self, patience: int, mode: str = 'min'):
        # TODO: store patience, mode, best (initialized to +/-inf), and a bad-step counter
        self.patience = patience 
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, metric: float) -> bool:
        # TODO: update best/counter, return True iff should stop
        if not self.best:
            self.best = metric
            return False 

        if self.mode == "min":
            if metric >= self.best:
                self.count += 1
            else:
                self.count = 0
                self.best = metric
        if self.mode == "max":
            if metric <= self.best:
                self.count += 1
            else:
                self.count = 0
                self.best = metric

        if self.count >= self.patience:
            return True   
        return False          
