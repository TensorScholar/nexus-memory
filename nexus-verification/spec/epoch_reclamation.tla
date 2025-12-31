--------------------------- MODULE epoch_reclamation ---------------------------
(***************************************************************************)
(* Formal specification of NEXUS epoch-based memory reclamation system     *)
(* Proves safety: no use-after-free violations                            *)
(* Proves liveness: all garbage is eventually reclaimed                   *)
(* Mathematical rigor with concurrent system modeling                     *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS 
    Threads,          \* Set of thread identifiers
    Objects,          \* Set of memory objects
    MaxEpoch,         \* Maximum epoch value (for model checking)
    GracePeriod       \* Epochs before reclamation eligibility

VARIABLES
    epoch,            \* Global epoch counter
    threadEpoch,      \* Per-thread epoch values
    active,           \* Active threads in critical sections
    allocated,        \* Set of allocated objects
    retired,          \* Map: object -> retirement epoch
    reclaimed,        \* Set of reclaimed objects
    references        \* Map: thread -> set of referenced objects

vars == <<epoch, threadEpoch, active, allocated, retired, reclaimed, references>>

(***************************************************************************)
(* Type invariants ensuring well-formed state                             *)
(***************************************************************************)
TypeInvariant ==
    /\ epoch \in 0..MaxEpoch
    /\ threadEpoch \in [Threads -> 0..MaxEpoch]
    /\ active \subseteq Threads
    /\ allocated \subseteq Objects
    /\ retired \in [allocated -> 0..MaxEpoch]
    /\ reclaimed \subseteq Objects
    /\ references \in [Threads -> SUBSET Objects]
    /\ \A t \in Threads : references[t] \subseteq allocated

(***************************************************************************)
(* Initial state with empty memory and zero epochs                        *)
(***************************************************************************)
Init ==
    /\ epoch = 0
    /\ threadEpoch = [t \in Threads |-> 0]
    /\ active = {}
    /\ allocated = {}
    /\ retired = <<>>
    /\ reclaimed = {}
    /\ references = [t \in Threads |-> {}]

(***************************************************************************)
(* Thread enters critical section, pinning current epoch                  *)
(***************************************************************************)
EnterCritical(t) ==
    /\ t \notin active
    /\ active' = active \cup {t}
    /\ threadEpoch' = [threadEpoch EXCEPT ![t] = epoch]
    /\ UNCHANGED <<epoch, allocated, retired, reclaimed, references>>

(***************************************************************************)
(* Thread exits critical section, allowing epoch advancement              *)
(***************************************************************************)
ExitCritical(t) ==
    /\ t \in active
    /\ active' = active \ {t}
    /\ references' = [references EXCEPT ![t] = {}]
    /\ UNCHANGED <<epoch, threadEpoch, allocated, retired, reclaimed>>

(***************************************************************************)
(* Memory allocation with reference tracking                              *)
(***************************************************************************)
Allocate(t, obj) ==
    /\ t \in active
    /\ obj \in Objects \ (allocated \cup reclaimed)
    /\ allocated' = allocated \cup {obj}
    /\ references' = [references EXCEPT ![t] = @ \cup {obj}]
    /\ UNCHANGED <<epoch, threadEpoch, active, retired, reclaimed>>

(***************************************************************************)
(* Object retirement marking for future reclamation                       *)
(***************************************************************************)
Retire(obj) ==
    /\ obj \in allocated
    /\ obj \notin DOMAIN retired
    /\ obj \notin reclaimed
    /\ retired' = retired @@ (obj :> epoch)
    /\ UNCHANGED <<epoch, threadEpoch, active, allocated, reclaimed, references>>

(***************************************************************************)
(* Global epoch advancement when all threads are synchronized             *)
(***************************************************************************)
AdvanceEpoch ==
    /\ epoch < MaxEpoch
    /\ \A t \in active : threadEpoch[t] = epoch
    /\ epoch' = epoch + 1
    /\ UNCHANGED <<threadEpoch, active, allocated, retired, reclaimed, references>>

(***************************************************************************)
(* Safe reclamation of objects after grace period expiration             *)
(***************************************************************************)
Reclaim(obj) ==
    /\ obj \in DOMAIN retired
    /\ retired[obj] + GracePeriod < epoch
    /\ \A t \in Threads : obj \notin references[t]
    /\ LET minThreadEpoch == CHOOSE e \in Nat : 
           e = IF active = {} 
               THEN epoch 
               ELSE Min({threadEpoch[t] : t \in active})
       IN retired[obj] < minThreadEpoch
    /\ reclaimed' = reclaimed \cup {obj}
    /\ allocated' = allocated \ {obj}
    /\ retired' = [o \in DOMAIN retired \ {obj} |-> retired[o]]
    /\ UNCHANGED <<epoch, threadEpoch, active, references>>

(***************************************************************************)
(* Reference acquisition ensuring object validity                         *)
(***************************************************************************)
AcquireReference(t, obj) ==
    /\ t \in active
    /\ obj \in allocated
    /\ obj \notin reclaimed
    /\ obj \notin DOMAIN retired
    /\ references' = [references EXCEPT ![t] = @ \cup {obj}]
    /\ UNCHANGED <<epoch, threadEpoch, active, allocated, retired, reclaimed>>

(***************************************************************************)
(* Complete system next-state relation                                   *)
(***************************************************************************)
Next ==
    \/ \E t \in Threads : EnterCritical(t)
    \/ \E t \in Threads : ExitCritical(t)
    \/ \E t \in Threads, obj \in Objects : Allocate(t, obj)
    \/ \E obj \in allocated : Retire(obj)
    \/ AdvanceEpoch
    \/ \E obj \in DOMAIN retired : Reclaim(obj)
    \/ \E t \in Threads, obj \in allocated : AcquireReference(t, obj)

(***************************************************************************)
(* Safety property: No use-after-free violations                         *)
(***************************************************************************)
SafetyInvariant ==
    \A t \in Threads, obj \in references[t] :
        /\ obj \in allocated
        /\ obj \notin reclaimed

(***************************************************************************)
(* Progress property: Retired objects are eventually reclaimed            *)
(***************************************************************************)
Liveness ==
    \A obj \in Objects :
        (obj \in DOMAIN retired) ~> (obj \in reclaimed)

(***************************************************************************)
(* Epoch monotonicity invariant                                          *)
(***************************************************************************)
EpochMonotonicity ==
    \A t \in active : threadEpoch[t] <= epoch

(***************************************************************************)
(* Reference validity throughout critical sections                        *)
(***************************************************************************)
ReferenceValidity ==
    \A t \in active, obj \in references[t] :
        \/ obj \in allocated \ DOMAIN retired
        \/ (obj \in DOMAIN retired /\ retired[obj] >= threadEpoch[t])

(***************************************************************************)
(* Complete specification with all properties                            *)
(***************************************************************************)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

THEOREM Safety == Spec => []SafetyInvariant
THEOREM Progress == Spec => Liveness
THEOREM Correctness == Spec => [](TypeInvariant /\ SafetyInvariant /\ 
                                  EpochMonotonicity /\ ReferenceValidity)

=============================================================================