--------------------------- MODULE RefinementMapping ---------------------------
EXTENDS EpochReclamation, TLC, Integers, Sequences, FiniteSets

CONSTANTS 
    ConcreteThreads, ConcreteMemory, MaxConcreteEpoch

VARIABLES
    implEpoch, implActive, implGarbage, implProtected, implMemState

TypeInvariantImpl == 
    /\ implEpoch \in [ConcreteThreads -> 0..MaxConcreteEpoch]
    /\ implActive \subseteq ConcreteThreads
    /\ implGarbage \in Seq(ConcreteMemory)
    /\ implProtected \in [ConcreteThreads -> SUBSET ConcreteMemory]
    /\ implMemState \in [ConcreteMemory -> {"allocated", "freed", "reclaimed"}]

ThreadMap(t) == CHOOSE at \in Threads : Hash(t) % Cardinality(Threads) = at
MemoryMap(m) == CHOOSE am \in Memory : Hash(m) % Cardinality(Memory) = am

EpochMap(implE) ==
    [t \in Threads |-> 
        LET concreteT == CHOOSE ct \in ConcreteThreads : ThreadMap(ct) = t
        IN implE[concreteT] % MaxEpoch]

ActiveMap(implA) == {ThreadMap(t) : t \in implA}

GarbageMap(implG) ==
    LET mappedSeq == [i \in 1..Len(implG) |-> MemoryMap(implG[i])]
    IN SelectSeq(mappedSeq, LAMBDA m: m \in Memory)

ProtectedMap(implP) ==
    [t \in Threads |->
        LET concreteT == CHOOSE ct \in ConcreteThreads : ThreadMap(ct) = t
        IN {MemoryMap(m) : m \in implP[concreteT]} \cap Memory]

StateCorrespondence ==
    /\ epoch = EpochMap(implEpoch)
    /\ active = ActiveMap(implActive)  
    /\ garbage = GarbageMap(implGarbage)
    /\ protected = ProtectedMap(implProtected)

RefinementWellDefined ==
    /\ \A t \in ConcreteThreads : ThreadMap(t) \in Threads
    /\ \A m \in ConcreteMemory : MemoryMap(m) \in Memory
    /\ \A t \in ConcreteThreads : implEpoch[t] < MaxConcreteEpoch

RefinedMemorySafety ==
    \A m \in ConcreteMemory :
        (implMemState[m] = "reclaimed") =>
            \A t \in ConcreteThreads : m \notin implProtected[t]

============================================================================