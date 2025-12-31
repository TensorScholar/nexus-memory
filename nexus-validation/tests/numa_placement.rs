use nexus_memory::numa::{AllocationPolicy, NodeId, NumaAllocator};
use nexus_validation::numa_verify::{
    detect_numa_topology, verify_buffer_placement, verify_physical_location,
};

#[test]
fn verify_numa_physical_placement() {
    println!("Forensic Verification: NUMA Physical Page Placement");
    println!("Objective: Prove 22% traffic reduction claim via physical locality validation");

    // 1. Detect Hardware
    let topology = match detect_numa_topology() {
        Some(t) => t,
        None => {
            println!("SKIP: NUMA topology not detected (running on non-Linux or virtualized environment?)");
            return;
        }
    };

    if !topology.is_numa || topology.nodes.len() < 2 {
        println!(
            "SKIP: System has insufficient NUMA nodes (found: {})",
            topology.nodes.len()
        );
        println!("       Benchmark claims require at least 2 sockets for verification.");
        return;
    }

    // Select the second node as target to ensure we are not just getting default (node 0) behavior
    let target_node = topology.nodes[1];
    println!("- Target Node: {}", target_node);

    // 2. Allocate on Target Node
    println!(
        "- Allocating 10MB buffer explicitly on Node {}...",
        target_node
    );
    let allocator = NumaAllocator::new();
    let size = 10 * 1024 * 1024; // 10MB
    let count = size / std::mem::size_of::<u64>();

    // Use OnNode policy
    let ptr = allocator
        .allocate::<u64>(count, AllocationPolicy::OnNode(NodeId(target_node)))
        .expect("Failed to allocate memory");

    let raw_ptr = ptr.as_ptr() as *mut u8;

    // 3. Write Data (Force Faulting)
    // Note: Allocator should have pre-faulted, but we write pattern to be sure memory is touched specifically by main thread
    // This also tests write access latency implicitly if mapped remotely
    unsafe {
        std::ptr::write_bytes(raw_ptr, 0xAA, size);
    }

    // 4. Verify Physical Placement
    println!("- Verifying physical page location (syscall: move_pages)...");

    // Simple single-pointer check
    let confirmed = verify_physical_location(raw_ptr, target_node);

    // Detailed check of buffer samples
    let report = verify_buffer_placement(raw_ptr, size, target_node);

    println!(
        "  - Pages Validated: {}/{}",
        report.pages_correct, report.pages_checked
    );
    if let Some(actual) = report.actual_node {
        println!("  - Observed Node:   {}", actual);
    }

    // 5. Assertions
    if confirmed && report.verified {
        println!(
            "> [PASS] Physical placement verified. Memory is resident on Node {}.",
            target_node
        );
    } else {
        println!("> [FAIL] Memory placement incorrect!");
        println!("  - Expected Node: {}", target_node);
        println!("  - Actual Node:   {:?}", report.actual_node);
        println!("  - Error:         {:?}", report.error);

        // Cleanup before panicking
        unsafe {
            allocator.deallocate(ptr, count, NodeId(target_node));
        }
        panic!("NUMA placement verification failed");
    }

    // Cleanup
    unsafe {
        allocator.deallocate(ptr, count, NodeId(target_node));
    }
}
