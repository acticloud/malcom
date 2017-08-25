# malcom
Early Stage cost model for mal statements
work in progress....

#Configuration files

## config/mal_blacklist.txt
List of all the mal instruction we do not wish to consider (defile,mvc etc...)

## config/{db}_stats.txt
For each different db we want to use, there must be a | separated file,

that contains the following statistics for each column,

min value, max value, count, unique, width.

#File Structure


#Basic Classes

##MalDictionary
The dictionary that holds all the instructions


##MalInstruction
Base class for bookkeeping all the mal instruction metadata()

##MalInstruction interface
All MalInstruction sub classes must satisfy the following interface:

```
interface MalInstruction {
  def argCnt()                               -> List<int>

  def approxArgCnt(traind: MalDictionary, G) -> List<int>

  def predictCount(traind: MalDictionary, G) -> List<Prediction>

  def kNN(traind: MalDictionary, k, G)       -> List<MalInstruction>
}
```
##MalArgument
