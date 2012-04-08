dts="0.1 0.05 0.025 0.01 0.005 0.0025 0.001 0.0005"

for dt in $dts; do
    echo "Solving with dt = $dt..."
    python primal.py dt=$dt | grep Functional
done
