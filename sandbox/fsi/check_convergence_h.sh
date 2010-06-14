nxs="40 60 80 100 120 140 160 180 200 220 240 260 280 300 320"

for nx in $nxs; do
    echo "Solving with nx = $nx..."
    python primal.py dt=$dt | grep Functional
done
