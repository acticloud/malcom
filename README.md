# malcom
Early Stage cost model for mal statements
work in progress....

## Example Usage: use the following sequence of functions to find the memory
##   footprint of a query

```
def predict_query_mem(train, test):
  blacklist = Utils.init_blacklist('black_list')

  cstats    = ColumnStats.fromFile('stats_file')

  traind    = MalDictionary.fromJsonFile(train, blacklist, cstats)

  testd     = MalDictionary.fromJsonFile(test, blacklist, cstats)

  pG        = testd.buildApproxGraph(traind)

  query_mem = testd.predictMaxMem(pG)
```

##TODO
1. Possible performance optimisation of the Malcom code: MalDictionary
   currently store all MAL instructions. This is necessary for the test query,
   but not for the training queries, because we don't need to keep the MAL
   instructions, e.g. bat.calc, sort, firstn,  which we can do direct
   prediction. Basically, we only need to keep the variable MAL instructions,
   such as SELECTs and JOINs.
2. fix mirror instruction memory footprint (is 0....)
3. fix substring memory error


# Configuration files

## config/mal_blacklist.txt
List of all the mal instruction we do not wish to consider (define,mvc etc...)

## config/{db}_stats.txt
For each different db we want to use, there must be a | separated file,

that contains the following statistics for each column (the order here
matters):

min value, max value, count, unique, width.

# File Structure
./src/malcom.py     : the main program

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
Base class for bookkeeping information for the different types of MAL
 instructions metadata

## MalInstruction interface
All MalInstruction sub classes must satisfy the following interface:

```
interface MalInstruction {
  def argCnt()                               -> List<int>

  def approxArgCnt(traind: MalDictionary, G) -> List<int>

  def predict(traind: MalDictionary, G) -> List<Prediction>

  def kNN(traind: MalDictionary, k, G)       -> List<MalInstruction>
}
```

## MalArgument
Containts structure to denote a MAL variable (i.e. those starting with a 'C' or an 'X')
Used by mal_instr.py

