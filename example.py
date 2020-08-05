from amp import Amp
from LithiumVacancyOrdering import VacancyOrdering


ocv = VacancyOrdering()
calc=Amp.load('calc.amp')

ocv.load_atoms('LCO_3x3.traj',calc) 


ocv.run_GCMC(T=300, num_sweeps=270, monoclinic=False)


ocv.get_hull(filename='energy_out.txt')

