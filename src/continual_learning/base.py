class BaseCLMethod:
    """Base class for all continual learning methods."""

    def __init__(self, model, scenario):
        self.model = model
        self.scenario = scenario

    def before_task(self, task_id):
        """Prepare for learning a new task."""
        pass

    def after_task(self, task_id):
        """Consolidate knowledge after learning a task."""
        pass

    def compute_loss(self, outputs, targets, task_id):
        """Compute task-specific loss with method-specific regularization."""
        import torch.nn.functional as F
        return F.cross_entropy(outputs, targets)