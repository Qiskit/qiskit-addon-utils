"""Script to visualize different combinations of pre/post selection passes."""

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    AddPreSelectionMeasures,
)


def create_test_circuit():
    """Create a non-trivial 10-qubit circuit."""
    qc = QuantumCircuit(10, 10)
    
    # Create some entanglement
    qc.h(0)
    qc.h(5)
    
    # Create GHZ-like states
    for i in range(4):
        qc.cx(0, i + 1)
    
    for i in range(4):
        qc.cx(5, i + 6)
    
    # Add some rotations
    qc.rz(0.5, 2)
    qc.rx(0.3, 7)
    qc.ry(0.7, 4)
    
    # Cross-entangle the two groups
    qc.cx(4, 6)
    
    # Measure all qubits
    qc.measure(range(10), range(10))
    
    return qc


def save_circuit(circuit, filename):
    """Save circuit diagram to file."""
    try:
        circuit.draw('mpl', filename=f'pics/{filename}', fold=-1)
        print(f"✓ Saved: pics/{filename}")
    except Exception as e:
        print(f"✗ Failed to save {filename}: {e}")


def main():
    """Generate visualizations for all pass combinations."""
    print("Creating test circuit...")
    base_circuit = create_test_circuit()
    
    # Create a coupling map for the passes that need it
    # Linear coupling for simplicity
    coupling_map = CouplingMap.from_line(10)
    
    print("\nGenerating circuit visualizations...\n")
    
    # 1. Original circuit
    save_circuit(base_circuit, "00_original_circuit.png")
    
    # 2. Post-selection only (default name)
    print("\n1. Post-selection only (default creg name)...")
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    circ = pm.run(base_circuit)
    save_circuit(circ, "01_post_sel_default.png")
    
    # 3. Post-selection only (custom name)
    print("\n2. Post-selection only (custom creg name)...")
    pm = PassManager([AddPostSelectionMeasures(
        x_pulse_type="rx",
        post_selection_suffix="_check"
    )])
    circ = pm.run(base_circuit)
    save_circuit(circ, "02_post_sel_custom.png")
    
    # 4. Pre-selection only (default name)
    print("\n3. Pre-selection only (default creg name)...")
    pm = PassManager([AddPreSelectionMeasures(
        coupling_map,
        x_pulse_type="rx"
    )])
    circ = pm.run(base_circuit)
    save_circuit(circ, "03_pre_sel_default.png")
    
    # 5. Pre-selection only (custom name)
    print("\n4. Pre-selection only (custom creg name)...")
    pm = PassManager([AddPreSelectionMeasures(
        coupling_map,
        x_pulse_type="rx",
        pre_selection_suffix="_init"
    )])
    circ = pm.run(base_circuit)
    save_circuit(circ, "04_pre_sel_custom.png")
    
    # 6. Both passes: pre first, post second (default names)
    print("\n5. Both passes: pre-sel first, post-sel second (default names)...")
    pm = PassManager([
        AddPreSelectionMeasures(coupling_map, x_pulse_type="rx"),
        AddPostSelectionMeasures(x_pulse_type="rx")
    ])
    circ = pm.run(base_circuit)
    save_circuit(circ, "05_pre_then_post_default.png")
    
    # 7. Both passes: post first, pre second (default names)
    print("\n6. Both passes: post-sel first, pre-sel second (default names)...")
    pm = PassManager([
        AddPostSelectionMeasures(x_pulse_type="rx"),
        AddPreSelectionMeasures(coupling_map, x_pulse_type="rx")
    ])
    circ = pm.run(base_circuit)
    save_circuit(circ, "06_post_then_pre_default.png")
    
    # 8. Both passes: pre first, post second (custom names)
    print("\n7. Both passes: pre-sel first, post-sel second (custom names)...")
    pm = PassManager([
        AddPreSelectionMeasures(
            coupling_map,
            x_pulse_type="rx",
            pre_selection_suffix="_init"
        ),
        AddPostSelectionMeasures(
            x_pulse_type="rx",
            post_selection_suffix="_check",
            ignore_creg_suffixes=["_init"]
        )
    ])
    circ = pm.run(base_circuit)
    save_circuit(circ, "07_pre_then_post_custom.png")
    
    # 9. Both passes: post first, pre second (custom names)
    print("\n8. Both passes: post-sel first, pre-sel second (custom names)...")
    pm = PassManager([
        AddPostSelectionMeasures(
            x_pulse_type="rx",
            post_selection_suffix="_check"
        ),
        AddPreSelectionMeasures(
            coupling_map,
            x_pulse_type="rx",
            pre_selection_suffix="_init",
            ignore_creg_suffixes=["_check"]  # Ignore the custom post-selection suffix
        )
    ])
    circ = pm.run(base_circuit)
    save_circuit(circ, "08_post_then_pre_custom.png")
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("Check the 'pics/' directory for the generated images.")
    print("="*60)


if __name__ == "__main__":
    main()

# Made with Bob
