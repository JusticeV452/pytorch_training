from pydantic import BaseModel

class ParamManager(BaseModel):
    """Base class to hold parameters with validation and default handling."""
    class Config:
        extra = "forbid"

    def __getitem__(self, item):
        """Allow dict-style access: obj['param']"""
        return getattr(self, item)

    def __setitem__(self, key, value):
        """Allow dict-style assignment: obj['param'] = val"""
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default=default)

    def as_dict(self):
        """Serialize all params to dict"""
        return self.model_dump()
    
    # Temporary backwards compatibility
    @property
    def params(self):
        return self
