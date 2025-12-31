--------------------------- MODULE EpochReclamation ---------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS 
    Threads,          \* Set of thread identifiers
    Objects,          \* Set of memory objects
    Paradigms,        \* Set of paradigms {Batch, Stream, Graph}
    MaxEpoch,         \* Maximum epoch value
    MaxGarbage        \* Maximum garbage list size

ASSUME /\ Threads # {}
       /\ Objects # {}
       /\ Paradigms # {}
       /\ MaxEpoch \in Nat /\ MaxEpoch > 0
       /\ MaxGarbage \in Nat /\ MaxGarbage > 0

VARIABLES
    globalEpoch,      \* Current global epoch
    threadEpoch,      \* threadEpoch[t] = epoch of thread t
    threadActive,     \* threadActive[t] = TRUE if thread t is active
    threadParadigm,   \* threadParadigm[t] = current paradigm of thread t
    paradigmEpoch,    \* paradigmEpoch[p] = local epoch for paradigm p
    paradigmActive,   \* paradigmActive[p] = set of active threads in paradigm p
    allocated,        \* Set of allocated objects
    referenced,       \* referenced[o] = set of threads referencing object o
    garbageList,      \* garbageList[e] = objects to reclaim when safe
    crossRefs,        \* crossRefs[o] = set of paradigms referencing object o
    freed,            \* Set of freed objects (history variable)
    accessHistory     \* History of object accesses for verification

vars == <<globalEpoch, threadEpoch, threadActive, threadParadigm,
          paradigmEpoch, paradigmActive, allocated, referenced, 
          garbageList, crossRefs, freed, accessHistory>>

TypeOK == 
    /\ globalEpoch \in 0..MaxEpoch
    /\ threadEpoch \in [Threads -> 0..MaxEpoch]
    /\ threadActive \in [Threads -> BOOLEAN]
    /\ threadParadigm \in [Threads -> Paradigms \cup {{}}]
    /\ paradigmEpoch \in [Paradigms -> 0..MaxEpoch]
    /\ paradigmActive \in [Paradigms -> SUBSET Threads]
    /\ allocated \subseteq Objects
    /\ referenced \in [Objects -> SUBSET Threads]
    /\ garbageList \in [0..MaxEpoch -> SUBSET Objects]
    /\ crossRefs \in [Objects -> SUBSET Paradigms]
    /\ freed \subseteq Objects
    /\ accessHistory \in [Objects -> Seq(Threads)]

Init ==
    /\ globalEpoch = 0
    /\ threadEpoch = [t \in Threads |-> 0]
    /\ threadActive = [t \in Threads |-> FALSE]
    /\ threadParadigm = [t \in Threads |-> {}]
    /\ paradigmEpoch = [p \in Paradigms |-> 0]
    /\ paradigmActive = [p \in Paradigms |-> {}]
    /\ allocated = {}
    /\ referenced = [o \in Objects |-> {}]
    /\ garbageList = [e \in 0..MaxEpoch |-> {}]
    /\ crossRefs = [o \in Objects |-> {}]
    /\ freed = {}
    /\ accessHistory = [o \in Objects |-> <<>>]

MinThreadEpoch == 
    IF \E t \in Threads : threadActive[t]
    THEN Min({threadEpoch[t] : t \in Threads \ {t \in Threads : threadActive[t]}})
    ELSE globalEpoch

MinParadigmEpoch ==
    IF \E p \in Paradigms : paradigmActive[p] # {}
    THEN Min({paradigmEpoch[p] : p \in Paradigms \ {p \in Paradigms : paradigmActive[p] = {}}})
    ELSE globalEpoch

SafeEpoch == Min({MinThreadEpoch, MinParadigmEpoch})

CanReclaim(epoch) == epoch < SafeEpoch

IsReferenced(obj) == \/ referenced[obj] # {} \/ crossRefs[obj] # {}

ThreadEnter(t, p) ==
    /\ ~threadActive[t]
    /\ threadActive' = [threadActive EXCEPT ![t] = TRUE]
    /\ threadEpoch' = [threadEpoch EXCEPT ![t] = globalEpoch]
    /\ threadParadigm' = [threadParadigm EXCEPT ![t] = p]
    /\ paradigmActive' = [paradigmActive EXCEPT ![p] = paradigmActive[p] \cup {t}]
    /\ UNCHANGED <<globalEpoch, paradigmEpoch, allocated, referenced, 
                   garbageList, crossRefs, freed, accessHistory>>

ThreadExit(t) ==
    /\ threadActive[t]
    /\ LET p == threadParadigm[t] IN
       /\ threadActive' = [threadActive EXCEPT ![t] = FALSE]
       /\ paradigmActive' = [paradigmActive EXCEPT ![p] = paradigmActive[p] \ {t}]
       /\ threadParadigm' = [threadParadigm EXCEPT ![t] = {}]
    /\ UNCHANGED <<globalEpoch, threadEpoch, paradigmEpoch, allocated, 
                   referenced, garbageList, crossRefs, freed, accessHistory>>

Allocate(t, obj) ==
    /\ threadActive[t]
    /\ obj \notin allocated
    /\ obj \notin freed
    /\ allocated' = allocated \cup {obj}
    /\ referenced' = [referenced EXCEPT ![obj] = {t}]
    /\ LET p == threadParadigm[t] IN
       crossRefs' = [crossRefs EXCEPT ![obj] = {p}]
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                  paradigmEpoch, paradigmActive, garbageList, freed, accessHistory>>

Access(t, obj) ==
    /\ threadActive[t]
    /\ obj \in allocated
    /\ obj \notin freed
    /\ referenced' = [referenced EXCEPT ![obj] = referenced[obj] \cup {t}]
    /\ LET p == threadParadigm[t] IN
       crossRefs' = [crossRefs EXCEPT ![obj] = crossRefs[obj] \cup {p}]
    /\ accessHistory' = [accessHistory EXCEPT ![obj] = Append(accessHistory[obj], t)]
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                   paradigmEpoch, paradigmActive, allocated, garbageList, freed>>

Release(t, obj) ==
    /\ threadActive[t]
    /\ t \in referenced[obj]
    /\ referenced' = [referenced EXCEPT ![obj] = referenced[obj] \ {t}]
    /\ LET p == threadParadigm[t] IN
       IF \A t2 \in Threads : (t2 # t /\ t2 \in referenced[obj]) => threadParadigm[t2] # p
       THEN crossRefs' = [crossRefs EXCEPT ![obj] = crossRefs[obj] \ {p}]
       ELSE crossRefs' = crossRefs
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                   paradigmEpoch, paradigmActive, allocated, garbageList, 
                   freed, accessHistory>>

Retire(t, obj) ==
    /\ threadActive[t]
    /\ obj \in allocated
    /\ ~IsReferenced(obj)
    /\ Cardinality(garbageList[threadEpoch[t]]) < MaxGarbage
    /\ allocated' = allocated \ {obj}
    /\ garbageList' = [garbageList EXCEPT ![threadEpoch[t]] = 
                       garbageList[threadEpoch[t]] \cup {obj}]
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                   paradigmEpoch, paradigmActive, referenced, crossRefs, 
                   freed, accessHistory>>

AdvanceGlobalEpoch ==
    /\ globalEpoch < MaxEpoch
    /\ globalEpoch' = globalEpoch + 1
    /\ UNCHANGED <<threadEpoch, threadActive, threadParadigm, paradigmEpoch,
                   paradigmActive, allocated, referenced, garbageList, 
                   crossRefs, freed, accessHistory>>

UpdateThreadEpoch(t) ==
    /\ threadActive[t]
    /\ threadEpoch[t] < globalEpoch
    /\ threadEpoch' = [threadEpoch EXCEPT ![t] = globalEpoch]
    /\ UNCHANGED <<globalEpoch, threadActive, threadParadigm, paradigmEpoch,
                   paradigmActive, allocated, referenced, garbageList, 
                   crossRefs, freed, accessHistory>>

UpdateParadigmEpoch(p) ==
    /\ paradigmActive[p] # {}
    /\ paradigmEpoch[p] < globalEpoch
    /\ \A t \in paradigmActive[p] : threadEpoch[t] >= paradigmEpoch[p]
    /\ paradigmEpoch' = [paradigmEpoch EXCEPT ![p] = 
                        Min({threadEpoch[t] : t \in paradigmActive[p]})]
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                   paradigmActive, allocated, referenced, garbageList, 
                   crossRefs, freed, accessHistory>>

Reclaim(epoch) ==
    /\ CanReclaim(epoch)
    /\ garbageList[epoch] # {}
    /\ freed' = freed \cup garbageList[epoch]
    /\ garbageList' = [garbageList EXCEPT ![epoch] = {}]
    /\ UNCHANGED <<globalEpoch, threadEpoch, threadActive, threadParadigm,
                   paradigmEpoch, paradigmActive, allocated, referenced, 
                   crossRefs, accessHistory>>

Next ==
    \/ \E t \in Threads, p \in Paradigms : ThreadEnter(t, p)
    \/ \E t \in Threads : ThreadExit(t)
    \/ \E t \in Threads, obj \in Objects : Allocate(t, obj)
    \/ \E t \in Threads, obj \in Objects : Access(t, obj)
    \/ \E t \in Threads, obj \in Objects : Release(t, obj)
    \/ \E t \in Threads, obj \in Objects : Retire(t, obj)
    \/ AdvanceGlobalEpoch
    \/ \E t \in Threads : UpdateThreadEpoch(t)
    \/ \E p \in Paradigms : UpdateParadigmEpoch(p)
    \/ \E e \in 0..(MaxEpoch-1) : Reclaim(e)

Spec == Init /\ [][Next]_vars

NoUseAfterFree ==
    \A obj \in freed : 
        /\ obj \notin allocated
        /\ referenced[obj] = {}
        /\ \A t \in Threads : threadActive[t] => 
             \A i \in DOMAIN accessHistory[obj] :
                accessHistory[obj][i] = t => 
                \E j \in 1..i : ~threadActive[accessHistory[obj][j]]

MemorySafety == \A e \in 0..MaxEpoch : \A obj \in garbageList[e] : ~IsReferenced(obj)

EpochMonotonicity ==
    /\ \A t \in Threads : threadEpoch[t] <= globalEpoch
    /\ \A p \in Paradigms : paradigmEpoch[p] <= globalEpoch

CrossParadigmIntegrity ==
    \A obj \in Objects : \A p \in crossRefs[obj] :
        \E t \in referenced[obj] : threadParadigm[t] = p

EventualReclamation ==
    \A obj \in Objects : \A e \in 0..MaxEpoch :
        obj \in garbageList[e] ~> (obj \in freed \/ IsReferenced(obj))

ThreadProgress ==
    \A t \in Threads : threadActive[t] ~> 
        (\E obj \in Objects : Allocate(t, obj) \/ Access(t, obj) \/ Release(t, obj) \/ ThreadExit(t))

WaitFree == \A t \in Threads : []<>(~threadActive[t] \/ \E obj \in Objects : ENABLED(Access(t, obj)))

SafetyInvariant == TypeOK /\ NoUseAfterFree /\ MemorySafety /\ EpochMonotonicity /\ CrossParadigmIntegrity
LivenessProperty == EventualReclamation /\ ThreadProgress /\ WaitFree

===========================================================================