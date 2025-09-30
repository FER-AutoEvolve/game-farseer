
class Result[TData]:
    '''
    A simple Result class that is used to return a value or an error message from a function.
    param `TData` - the type of the value that is returned on success; use `Result[Unit]` for logical result
    '''
    def __init__(self, value: TData | None, is_success: bool, message: str = "Operation successful"):
        self.is_success = is_success
        self.message = message
        self.value = value

    @classmethod
    def ok(cls, value: TData | None):
        '''
        Create a new Result instance with a success value.
        '''
        if TData is Unit and value is None: # if the result is logical, no need to provide a value
            return cls(None, True)
        elif TData is not Unit and value is None: # if the result is a value, a value must be provided
            return cls(value, False, "None data provided")
        else:
            return cls(value, True)


    @classmethod
    def err(cls, message: str = None):
        '''
        Create a new Result instance with an error message.
        '''
        return cls(None, False, message)

    def is_ok(self):
        '''
        Check if the result is a success.
        '''
        return self.is_success

    def is_err(self):
        '''
        Check if the result is an error.
        
        '''
        return not self.is_success

    def unwrap(self) -> TData:
        '''
        Get the value from the result. If the result is an error, raise a ValueError.
        '''
        if not self.is_success:
            raise ValueError("Called `unwrap` on an Err value: " + str(self.value))
        return self.value

class Unit:
    '''
    A simple class that represents nothing.
    '''
    def __repr__(self):
        return "Unit"
    
    def __eq__(self, other):
        return isinstance(other, Unit)