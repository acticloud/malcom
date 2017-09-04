# malcom
Early Stage cost model for mal statements
work in progress....

## Example Usage: Find the memory footprint of a query

```
def predict_query_mem(train, test):
  blacklist = Utils.init_blacklist('black_list')

  cstats    = ColumnStats.fromFile('stats_file')

  traind    = MalDictionary.fromJsonFile(train, blacklist, cstats)

  testd     = MalDictionary.fromJsonFile(test, blacklist, cstats)

  pG        = testd.buildApproxGraph(traind)

  query_mem = d2.predictMaxMem(pG)
```


# Configuration files

## config/mal_blacklist.txt
List of all the mal instruction we do not wish to consider (define,mvc etc...)

## config/{db}_stats.txt
For each different db we want to use, there must be a | separated file,

that contains the following statistics for each column,

min value, max value, count, unique, width.

# File Structure
./src/malcom.py     : the driver program

./src/mal_dict.py   : mal dictionary stuff

./src/mal_instr.py  : mal instruction bookkeeping

./src/mal_arg.py    : mal argument class

./src/stats.py      : column statistics

./src/experiments.py: some experiments

./src/utils.py      : just utilities

# Basic Classes

## MalDictionary
The dictionary that holds all the instructions


## MalInstruction
Base class for bookkeeping all the mal instruction metadata()

## MalInstruction interface
All MalInstruction sub classes must satisfy the following interface:

```
interface MalInstruction {
  def argCnt()                               -> List<int>

  def approxArgCnt(traind: MalDictionary, G) -> List<int>

  def predictCount(traind: MalDictionary, G) -> List<Prediction>

  def kNN(traind: MalDictionary, k, G)       -> List<MalInstruction>
}
```

## MalArgument
