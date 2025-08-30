"""
This was a friction point I initially thought was a bug, but ended up just
being a misunderstanding of how to use the class-based UDF syntax.

I forgot to add the init args using the .with_init_args() method. 

I do think there are opportunities to improve the error messages here. 

"""
import daft
from daft import UDF

@daft.udf(return_dtype=daft.DataType.string())
class MyUdf:
    def __init__(self, text="world", foo="bar"):
        self.text = text
        self.foo = foo

    def __call__(self, arg1: daft.Series, arg2: daft.Series):
        arg1 = arg1.to_pylist()
        arg2 = arg2.to_pylist()
        return [arg1 + self.text + arg2 + self.foo for arg1, arg2 in zip(arg1, arg2)]   

    
df = daft.from_pylist([{
    "arg1": "world",
    "arg2": "something else"
}])

# %% Naive Expectation:
my_udf = MyUdf(text="hello", foo="goodbye") # < ---TypeError: missing a required argument: 'arg1'

df_result = df.with_column("result", my_udf(
    arg1=df["arg1"],
    arg2=df["arg2"]
))
df_result.show()


# Current Correct Usage: 
df_result = df.with_column("result", MyUdf.with_init_args(text="hello", foo="goodbye")(
    arg1=df["arg1"],
    arg2=df["arg2"]
))
df_result.show()
