"""Script to visualize edge-based post-selection with spectator measurements."""

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    AddPreSelectionMeasures,
    AddSpectatorMeasures,
    AddSpectatorMeasuresPreSelection,
)

try:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    USE_FAKE_BACKEND = True
except ImportError:
    from qiskit.transpiler import CouplingMap
    USE_FAKE_BACKEND = False


def transpile_to_backend(circuit, backend):
    """Transpile circuit to backend with trivial layout."""
    num_qubits = circuit.num_qubits
    return transpile(
        circuit,
        backend=backend,
        optimization_level=0,
        initial_layout=list(range(num_qubits))
    )


def create_edge_test_circuit():
    """Create a circuit suitable for edge-based post-selection.
    
    This circuit uses qubits 0-4 for computation.
    """
    # Create circuit with 5 qubits and 5 classical bits
    qc = QuantumCircuit(5, 5)
    
    # Create entanglement on active qubits (0-4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    
    # Add some rotations
    qc.rz(0.5, 1)
    qc.rx(0.3, 2)
    qc.ry(0.7, 3)
    
    # Measure only the active qubits (0-4)
    qc.measure(range(5), range(5))
    
    return qc


def create_edge_test_circuit_without_boxes():
    """Create a circuit without boxes for later boxing.
    
    This circuit uses qubits 0-4 for computation.
    """
    # Create circuit with only the data qubits (5) and 5 classical bits for measurements
    qc = QuantumCircuit(5, 5)
    
    # Create entanglement on active qubits (0-4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    
    # Add some rotations
    qc.rz(0.5, 1)
    qc.rx(0.3, 2)
    qc.ry(0.7, 3)
    
    # Measure only the active qubits (0-4)
    qc.measure(range(5), range(5))
    
    return qc


def save_circuit(circuit, filename, backend=None):
    """Save circuit diagram to file."""
    try:
        # Transpile to backend if provided
        if backend is not None:
            circuit = transpile_to_backend(circuit, backend)
        
        # Draw with idle_wires=False to hide unused qubits
        circuit.draw('mpl', filename=f'pics/edge/{filename}', fold=-1, idle_wires=False)
        print(f"✓ Saved: pics/edge/{filename}")
    except Exception as e:
        print(f"✗ Failed to save {filename}: {e}")


def main():
    """Generate visualizations for edge-based post-selection with spectators."""
    print("Creating edge-based test circuit...")
    base_circuit = create_edge_test_circuit()
    
    # Get coupling map and backend
    if USE_FAKE_BACKEND:
        print("Using FakeFez backend coupling map...")
        backend = FakeFez()
        coupling_map = backend.coupling_map
        
        # Transpile base circuit to backend FIRST to get all qubits
        print("Transpiling to FakeFez backend...")
        base_circuit_transpiled = transpile(
            base_circuit,
            backend=backend,
            optimization_level=0,
            initial_layout=list(range(5))
        )
    else:
        print("Using linear coupling map...")
        backend = None
        from qiskit.transpiler import CouplingMap
        coupling_map = CouplingMap.from_line(10)
        base_circuit_transpiled = base_circuit
    
    print("\nGenerating edge-based circuit visualizations...\n")
    
    # 1. Original circuit
    print("0. Original circuit...")
    save_circuit(base_circuit_transpiled, "00_original_circuit.png")
    
    # 2. Post-selection only (node-based)
    print("\n1. Node-based post-selection only...")
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "01_edge_post_sel.png")
    
    # 3. Post-selection + Spectator measurements
    print("\n2. Node-based post-selection + edge-based spectator measurements...")
    pm = PassManager([
        AddPostSelectionMeasures(x_pulse_type="rx"),
        AddSpectatorMeasures(coupling_map)
    ])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "02_edge_post_sel_with_spectators.png")
    
    # 4. Pre-selection only
    print("\n3. Pre-selection only...")
    pm = PassManager([AddPreSelectionMeasures(coupling_map, x_pulse_type="rx")])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "03_edge_pre_sel.png")
    
    # 5. Pre-selection + Spectator pre-selection measurements
    print("\n4. Pre-selection + spectator pre-selection measurements...")
    pm = PassManager([
        AddPreSelectionMeasures(coupling_map, x_pulse_type="rx"),
        AddSpectatorMeasuresPreSelection(coupling_map, x_pulse_type="rx")
    ])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "04_edge_pre_sel_with_spectators.png")
    
    # 6. Full stack: Pre-selection + Spectator pre-selection + Post-selection + Spectator post-selection
    print("\n5. Full stack: all passes combined...")
    pm = PassManager([
        AddPreSelectionMeasures(coupling_map, x_pulse_type="rx"),
        AddSpectatorMeasuresPreSelection(coupling_map, x_pulse_type="rx"),
        AddPostSelectionMeasures(x_pulse_type="rx"),
        AddSpectatorMeasures(coupling_map)
    ])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "05_edge_full_stack.png")
    
    # 7. Alternative order: Post-selection first, then pre-selection
    print("\n6. Alternative order: post-selection first, then pre-selection...")
    pm = PassManager([
        AddPostSelectionMeasures(x_pulse_type="rx"),
        AddSpectatorMeasures(coupling_map),
        AddPreSelectionMeasures(coupling_map, x_pulse_type="rx"),
        AddSpectatorMeasuresPreSelection(coupling_map, x_pulse_type="rx")
    ])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "06_edge_post_first_then_pre.png")
    
    # 8. Custom suffixes with spectators
    print("\n7. Custom suffixes with spectators...")
    pm = PassManager([
        AddPreSelectionMeasures(
            coupling_map,
            x_pulse_type="rx",
            pre_selection_suffix="_init"
        ),
        AddSpectatorMeasuresPreSelection(
            coupling_map,
            x_pulse_type="rx",
            spectator_creg_name="spec_init",
            pre_selection_suffix="_init"
        ),
        AddPostSelectionMeasures(
            x_pulse_type="rx",
            post_selection_suffix="_check",
            ignore_creg_suffixes=["_init", "spec_init"]
        ),
        AddSpectatorMeasures(
            coupling_map,
            ignore_creg_suffixes=["_init", "spec_init"],
            post_selection_suffix="_check"
        )
    ])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "07_edge_custom_suffixes.png")
    
    # 9. Only spectator measurements (no pre/post selection)
    print("\n8. Only edge-based spectator measurements...")
    pm = PassManager([AddSpectatorMeasures(coupling_map)])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "08_edge_spectators_only.png")
    
    # 10. Only spectator pre-selection measurements
    print("\n9. Only spectator pre-selection measurements...")
    pm = PassManager([AddSpectatorMeasuresPreSelection(coupling_map, x_pulse_type="rx")])
    circ = pm.run(base_circuit_transpiled)
    save_circuit(circ, "09_edge_spectators_pre_sel_only.png")
    
    # 11. Test 06 with boxes (Twirl annotations)
    print("\n10. Alternative order with Twirl boxes: post-selection first, then pre-selection...")
    try:
        from samplomatic.transpiler import generate_boxing_pass_manager
        
        # Step 1: Create circuit without boxes (only data qubits)
        base_circuit_no_boxes = create_edge_test_circuit_without_boxes()
        
        # Step 2: Transpile to backend layout to add necessary qubits
        if USE_FAKE_BACKEND:
            num_data_qubits = base_circuit_no_boxes.num_qubits
            preset_pm = generate_preset_pass_manager(
                optimization_level=0,
                backend=backend,
                initial_layout=list(range(num_data_qubits))
            )
            base_circuit_transpiled = preset_pm.run(base_circuit_no_boxes)
        else:
            base_circuit_transpiled = base_circuit_no_boxes
        
        # Step 3: Use samplomatic to add boxes
        boxing_pm = generate_boxing_pass_manager()
        base_circuit_with_boxes = boxing_pm.run(base_circuit_transpiled)
        
        # Step 4: Apply same passes as test 06
        pm = PassManager([
            AddPostSelectionMeasures(x_pulse_type="rx"),
            AddSpectatorMeasures(coupling_map),
            AddPreSelectionMeasures(coupling_map, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(coupling_map, x_pulse_type="rx")
        ])
        circ = pm.run(base_circuit_with_boxes)
        
        # Save without additional backend transpilation
        save_circuit(circ, "10_edge_post_first_then_pre_with_boxes.png", backend=None)
    except ImportError as e:
        print(f"⊘ Skipped: {e}")
    
    print("\n" + "="*60)
    print("All edge-based visualizations complete!")
    print("Check the 'pics/edge/' directory for the generated images.")
    print("="*60)


if __name__ == "__main__":
    main()

# Made with Bob