import numpy as np
from ase.optimize import BFGS
from ase.eos import EquationOfState
from fractions import Fraction
import sys
from LayeredOxideClusterExpansion import MonteCarlo, interactions #from the old ClusterExpansion code
from ase.io import read
import math
import random
from numpy.random import rand
from ase import Atoms


class VacancyOrdering:
    """
    A framework to generate or read in fully lithiated NMC structure and model the Li-vacancy ordering.
    Future would be to generate OCV from a given composition using either Grand Canonical or Canonical MC.

    Parameters
    ----------

    """


    def __init__(self):
        pass


    def load_atoms(self, atoms_file, calc, optimize=False,monoclinic=False):
        """
        Loads an ASE Atoms object from a file and then relaxes with the Amp calculator
        The goal is to create self.atoms that is relaxed before vacancies are formed. 

        Parameters
        ----------  
        atoms_file : :class:`str`
            File location of the atoms to read in. Can be a .cif, .traj, .gpw, .txt from GPAW output etc. 
        calc : :class:`Calculator <ase.calculators.calculator.Calculator>`
            Can be any calculator that inherits the ASE Calculator class: AMP, AMPtorch, comorant.ase, etc. But it must have forces!
       """
        #Must be fully lithiated structures
        #sys.exit('Not fully implemented')
        self.calc = calc
        self.atoms = read(atoms_file)
        self.atoms.set_calculator(self.calc)
        cell = self.atoms.get_cell()
        if monoclinic:
            self.x = round(cell[0,0]/(math.sqrt(3)*2.8),0)
            self.y = round(np.linalg.norm(cell[1])/2.8,0)
            self.z = round(cell[0,0]/13.2,0)
        else:
            self.x = round(cell[0,0]/2.8,0)
            self.y = round(np.linalg.norm(cell[1])/2.8,0)
            self.z = round(cell[0,0]/13.2,0)
        #print(self.x,self.y,self.z)
        #self.num_site = 
        if optimize:
            self.atoms.set_calculator(self.calc)
            self.optimize_struct(step=0.02, N_step=7, relax=True, fmax=0.5,monoclinic=monoclinic)
            print(self.atoms.cell)
            self.optimize_struct(step=0.02, N_step=7, relax=True, fmax=0.5,monoclinic=monoclinic)
            print(self.atoms.cell)


    def optimize_struct(self, step=0.01, N_step=5, relax=True, fmax=None, maxstep=None, T=0.1, relax_steps=10, monoclinic=False):
        cell = self.atoms.get_cell()
        name = self.atoms.get_chemical_formula(mode='hill')
        a = cell[0,0] #original lattice parameters
        b = cell[1,0]
        c = cell[1,1]
        sep = cell[2,2]
        step = step
        
        vol = self.atoms.get_volume()
        if monoclinic:
            volumes =[]
            energies=[]

            for x in np.linspace(1-step,1+step,5):
                cell[0,0] = a*x
                self.atoms.set_cell(cell, scale_atoms=True)
                energies.append(self.atoms.get_potential_energy())
                volumes.append(self.atoms.get_volume())
            eos = EquationOfState(volumes, energies)
            try:
                v0,e0,B= eos.fit()
            except:
                e, idx = min((val, idx) for (idx, val) in enumerate(energies))
                v0 = volumes[idx]
            
            cell[0,0]=a*(v0/vol)**Fraction('1/3')
            self.atoms.set_cell(cell, scale_atoms=True)
            vol = v0


            volumes =[]
            energies=[]
            for x in np.linspace(1-step,1+step,5):
                cell[1,0] = b*x
                cell[1,1] = c*x
                self.atoms.set_cell(cell, scale_atoms=True)
                energies.append(self.atoms.get_potential_energy())
                volumes.append(self.atoms.get_volume())
            eos = EquationOfState(volumes, energies)
            try:
                v0,e0,B= eos.fit()
            except:
                e, idx = min((val, idx) for (idx, val) in enumerate(energies))
                v0 = volumes[idx]
            cell[1,0]=b*(v0/vol)**Fraction('1/3')
            cell[1,1]=c*(v0/vol)**Fraction('1/3')

            self.atoms.set_cell(cell, scale_atoms=True)
            vol = v0

            area=vol/sep
            volumes =[]
            energies=[]
            for x in np.linspace(1-2*step,1+2*step,N_step):
                cell[2,2]=sep*x
                self.atoms.set_cell(cell, scale_atoms=True)
                energies.append(self.atoms.get_potential_energy())
                volumes.append(self.atoms.get_volume())
            eos = EquationOfState(volumes, energies)
            try:
                v0,e0,B= eos.fit()
                if not min(volumes) <= v0 <= max(volumes):
                    raise ValueError('Fit volume out of tested volume ranges')
            except:
                e, idx = min((val, idx) for (idx, val) in enumerate(energies))
                v0 = volumes[idx]
            cell[2,2] = v0/area
            self.atoms.set_cell(cell,scale_atoms=True)


        else:

            area=vol/sep
            volumes =[]
            energies=[]
            for x in np.linspace(1-2*step,1+2*step,N_step):
                cell[2,2]=sep*x
                self.atoms.set_cell(cell, scale_atoms=True)
                energies.append(self.atoms.get_potential_energy())
                volumes.append(self.atoms.get_volume())
            eos = EquationOfState(volumes, energies)
            try:
                v0,e0,B= eos.fit()
                if not min(volumes) <= v0 <= max(volumes):
                    raise ValueError('Fit volume out of tested volume ranges')
            except:
                e, idx = min((val, idx) for (idx, val) in enumerate(energies))
                v0 = volumes[idx]
            cell[2,2] = v0/area
            self.atoms.set_cell(cell,scale_atoms=True)
            vol = v0

            volumes =[]
            energies=[]
            for x in np.linspace(1-2*step,1+2*step,N_step):
                cell[0,0]=a*x
                cell[1,0]=b*x
                cell[1,1]=c*x
                self.atoms.set_cell(cell, scale_atoms=True)
                energies.append(self.atoms.get_potential_energy())
                volumes.append(self.atoms.get_volume())
            eos = EquationOfState(volumes, energies)
            try:
                v0,e0,B= eos.fit()
            except:
                e, idx = min((val, idx) for (idx, val) in enumerate(energies))
                v0 = volumes[idx]
            cell[0,0]=a*(v0/vol)**Fraction('1/2')
            cell[1,0]=b*(v0/vol)**Fraction('1/2')
            cell[1,1]=c*(v0/vol)**Fraction('1/2')
        
            self.atoms.set_cell(cell,scale_atoms=True)
    
        if relax:
            if fmax==None:
                fmax=0.55
            if maxstep==None:
                maxstep=0.01
            dyn=BFGS(atoms=self.atoms,trajectory=name+'.traj',logfile = name+'.log',maxstep=maxstep)
            dyn.run(fmax=fmax, steps=relax_steps)



    def run_GCMC(self, T, mu=0, num_sweeps=1000, num_vacancies=0, P=0, convergence = 0.001, monoclinic=False, initialize = True):
        kT= T*8.6173303*10**(-5) #eV
        kT_J = T*1.380649*10**(-23) #kg*m^2/s^2
        h = 6.62607004 * 10**(-34) #m^2 kg/s
        m = 1.1525801 * 10**(-26) #kg
        #h = 4.135667696*10**(-15) #eV/s
        mu_Li = -5.110803619 - mu  #eV #this sign conventions means a positive potential of mu Volts is applied #uncorrected mu_Li = -5.592988
        print(mu_Li)
        Lambda = h /math.sqrt(2*math.pi*kT_J*m) # in m
        Lambda *= 10**10 #Ang
        #print('Lambda', Lambda)
        self.r_Li = 1.82 # 0.92 #angstrom
        sym = self.atoms.get_chemical_symbols()


        if initialize:
            self.N = len([atom for atom in sym if atom=='Li']) #Number of MC spots
            self.initialize_vacancy(num_vacancies, relax=False, fmax=0.55)
        
            '''
            cell = self.atoms.get_cell()
            self.optimize_struct(step=0.01, N_step=7, relax=True, fmax=0.55, relax_steps=10) #for good measure?

            x_a = self.atoms.cell[0,0]/cell[0,0]
            x_c = self.atoms.cell[2,2]/cell[2,2]

            self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2]
            self.Va_pos[:,2] = x_c*self.Va_pos[:,2]
            '''
        
            self.energy = self.atoms.get_potential_energy()
            self.energy_vals = [self.energy]
            
            self.c = [self.atoms.cell[2,2]/self.z]
            self.a = [self.atoms.cell[0,0]/self.x]

        
        self.pos = self.atoms.get_positions()

        self.N_Li = len(self.Li_coords)
        self.compositions = [self.N_Li] #number of Li's 

        self.V = self.atoms.get_volume() - self.N_Li * (4/3) * math.pi * self.r_Li**3
        #self.volumes = [self.V]
        #if monoclinic:
        #    self.angles = [120]
        
        n=0
        dE = 100
        #for n in range(num_sweeps):
        while n<num_sweeps and (dE > convergence or self.N_Li<self.N):
            changed = False
            start_E = self.energy
            if (rand() <= 0.5): # or self.N_Li==0) and self.N_Li != self.N:
                #add a Li
                if self.N_Li == self.N:
                    pass
                else:
                    i = random.choice(range(len(self.Va_pos)))
                    li_atom = Atoms('Li', positions=[self.Va_pos[i]])
                    atoms2 = self.atoms.__add__(li_atom)
                    atoms2.set_calculator(self.calc)
                    energy_new = atoms2.get_potential_energy()
                    delta_E = energy_new - self.energy - mu_Li
                    if delta_E <= 0 or (self.V/(Lambda**3 * (self.N_Li+1)) * np.exp(-(delta_E/kT)) >= rand()):
                        changed = True
                        #print(delta_E)
                        #print(self.V/(Lambda**3 * (self.N_Li+1)) * np.exp(-(delta_E/kT)))
                        #keep addition
                        self.atoms=atoms2.copy()
                        self.atoms.set_calculator(self.calc)
                        self.Li_coords.append(len(self.atoms)-1)
                        self.Va_pos = np.delete(self.Va_pos, i, axis=0)
                        self.energy = energy_new
                        self.pos = self.atoms.get_positions()
                        self.V += -(4/3) * math.pi * self.r_Li**3
                        self.N_Li += 1
                
            else:
                #remove a Li
                if self.N_Li==0:
                    pass
                else:
                    i = random.choice(self.Li_coords)
                    atoms2 = self.atoms.copy()
                    Li_pos=self.pos[i].copy() 
                    del atoms2[i]
                    atoms2.set_calculator(self.calc)
                    energy_new = atoms2.get_potential_energy()
                    delta_E = energy_new - self.energy + mu_Li
                    if delta_E <= 0 or ((self.N_Li*Lambda**3) /self.V * np.exp(-(delta_E/kT)) >= rand()):
                        changed = True
                        #keep removal
                        self.atoms=atoms2.copy()
                        self.atoms.set_calculator(self.calc)
                        self.Va_pos = np.append(self.Va_pos, Li_pos.reshape(1,3), axis=0)
                        sym = self.atoms.get_chemical_symbols()
                        self.Li_coords = []
                        for i, atom in enumerate(sym):
                            if atom=='Li':
                                self.Li_coords.append(i)
                                #self.Li_coords.remove(i)
                        self.energy = energy_new
                        self.pos = self.atoms.get_positions()
                        self.N_Li += -1
                    
                        #if self.N_Li != len(self.Li_coords):
                        #    print('N_Li fail')
                        self.V += (4/3) * math.pi * self.r_Li**3
                if self.energy != self.atoms.get_potential_energy():
                    print('GCMC fail')
            if not (self.N_Li == self.N or self.N_Li==0):
                for s in range(self.N):
                    #print(s)
                    self.sweep(kT, volume_change=True, P=P)

            if changed:
                cell = self.atoms.get_cell()
                
                self.optimize_struct(step=0.01, N_step=5, relax=False, fmax=0.25, relax_steps=2, monoclinic=monoclinic)

                x_a = self.atoms.cell[0,0]/cell[0,0]
                x_c = self.atoms.cell[2,2]/cell[2,2]

                self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2]
                self.Va_pos[:,2] = x_c*self.Va_pos[:,2]

            self.energy = self.atoms.get_potential_energy()
            self.V = self.atoms.get_volume() - self.N_Li * (4/3) * math.pi * self.r_Li**3
            self.pos = self.atoms.get_positions()


            self.energy_vals.append(self.energy)
            #self.V = self.atoms.get_volume()
            #self.volumes.append(self.V)
            self.c.append(self.atoms.cell[2,2]/self.z)
            self.a.append(self.atoms.cell[0,0]/self.x)
            self.compositions.append(self.N_Li)
            #if abs(self.energy-self.atoms.get_potential_energy())>0.001:
            #    print('fails')
            if monoclinic:
                self.angles.append(self.beta)
                print(self.energy, self.N_Li, self.atoms.cell[0,0], self.atoms.cell[2,2], self.beta)
            else:
                print(self.energy, self.N_Li, self.atoms.cell[0,0], self.atoms.cell[2,2])
            n += 1
            dE = abs(self.energy - start_E)
            #self.atoms.write('xyz/out'+str(n)+'.xyz')
            #print('dE is ', dE) 

    def sweep(self, kT, volume_change=False, P=0):
        if self.N_Li>0 and len(self.Va_pos)>0:
            #self.pos = self.atoms.get_positions()
            i = random.choice(self.Li_coords)
            j = random.choice(range(len(self.Va_pos)))
            #swap positions
            Li_pos=self.pos[i].copy()
            self.pos[i]=self.Va_pos[j].copy()
            self.Va_pos[j] = Li_pos
            self.atoms.set_positions(self.pos)
            
            energy_new=self.atoms.get_potential_energy()
            delta_E = energy_new - self.energy
            #r = rand()
            if (delta_E <= 0) or (rand() <= np.exp(-(delta_E/kT))):
                #print(delta_E, r)
                #keep change
                #print('keeping change and delta_E is ', delta_E)
                self.energy = energy_new
                #print(energy)
            else:
                #reject
                #swap positions back
                self.Va_pos[j] = self.pos[i].copy()
                self.pos[i] = Li_pos.copy()
                self.atoms.set_positions(self.pos)

    def initialize_vacancy(self, n_Va, relax=False, fmax=None):

        sym = self.atoms.get_chemical_symbols()
        self.Li_coords = []
        for i, atom in enumerate(sym):
            if atom=='Li':
                self.Li_coords.append(i)
            
        self.pos = self.atoms.get_positions()
        self.Va_pos = []

        for deletings in range(n_Va):
            i = random.choice(self.Li_coords)
            self.Va_pos.append(self.pos[i])
            del self.atoms[i]
            sym = self.atoms.get_chemical_symbols()
            self.pos = self.atoms.get_positions()
            self.Li_coords = []
            for i, atom in enumerate(sym):
                if atom=='Li':
                    self.Li_coords.append(i)

        self.Va_pos = np.array(self.Va_pos)

        cell = self.atoms.get_cell()
        self.optimize_struct(step=0.02, N_step=7, relax=relax, fmax=fmax if fmax is not None else 0.55, relax_steps=10) 
        
        x_a = self.atoms.cell[0,0]/cell[0,0]
        x_c = self.atoms.cell[2,2]/cell[2,2]

        if n_Va >0:
            self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2]
            self.Va_pos[:,2] = x_c*self.Va_pos[:,2]

            


    def get_hull(self, filename='out.txt'):
        hull = np.zeros(self.N+1)
        lattice_a = np.zeros(self.N+1)
        lattice_c = np.zeros(self.N+1)
        for i, N_Li in enumerate(self.compositions):
            if self.energy_vals[i] < hull[N_Li]:
                hull[N_Li] = self.energy_vals[i]
                lattice_a[N_Li] = self.a[i]
                lattice_c[N_Li] = self.c[i]
        comp = np.zeros(self.N+1)
        file = open(filename,'w')
        for i in range(self.N+1):
            comp[i] = i/float(self.N)
            file.write(str(comp[i])+' '+str(hull[i])+' '+str(lattice_a[i])+' '+str(lattice_c[i])+'\n')
        file.close()
            

    def swap(self,relax=False):
        i = random.choice(self.Li_coords)
        j = random.choice(range(len(self.Va_pos)))
        
        #swap positions
        Li_pos=self.pos[i].copy()
        self.pos[i]=self.Va_pos[j]
        self.Va_pos[j]= Li_pos
        self.atoms.set_positions(self.pos)

        #if relax:
        cell = self.atoms.get_cell()  
        self.optimize_struct(step=0.02, N_step=3, relax=relax, fmax=0.5, relax_steps=2)
        x_a = self.atoms.cell[0,0]/cell[0,0]
        x_c = self.atoms.cell[2,2]/cell[2,2]
        self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2] 
        self.Va_pos[:,2] = x_c*self.Va_pos[:,2] 
            
        self.energy=self.atoms.get_potential_energy()
        self.energy_vals.append(self.energy)





    def run_CMC(self, T, num_sweeps=1000, num_vacancies = 0, convergence = 0.001, monoclinic=False,relax=False):
        kT= T*8.6173303*10**(-5) #eV
        #h = 4.135667696*10**(-15) #eV/s
        #mu_Li = -5.592988 #eV
        #Lambda = h /math.sqrt(2*math.pi*kT)
        #self.r_Li = 1.82 # 0.92 #angstrom
        sym = self.atoms.get_chemical_symbols()

        self.N = len([atom for atom in sym if atom=='Li']) #Number of MC spots
        self.initialize_vacancy(num_vacancies, relax=False, fmax=0.55)
        
        '''
        cell = self.atoms.get_cell()
        self.optimize_struct(step=0.01, N_step=7, relax=False, fmax=0.55, relax_steps=10) #for good measure?

        x_a = self.atoms.cell[0,0]/cell[0,0]
        x_c = self.atoms.cell[2,2]/cell[2,2]

        self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2]
        self.Va_pos[:,2] = x_c*self.Va_pos[:,2]
        '''

        self.energy = self.atoms.get_potential_energy()
        self.energy_vals = [self.energy]

        #self.c = [self.atoms.cell[2,2]/self.z]
        #self.a = [self.atoms.cell[0,0]/self.x]
        #if monoclinic:
        #    self.b = [np.linalg.norm(self.atoms.cell[1])/self.x]
        
        self.pos = self.atoms.get_positions()

        #self.N_Li = len(self.Li_coords)
        #self.compositions = [self.N_Li] #number of Li's                                                                                                                                                         

        #self.V = self.atoms.get_volume() - self.N_Li * (4/3) * math.pi * self.r_Li**3

        n=0
        dE = 100
        #for n in range(num_sweeps):
        self.N_Li = len(self.Li_coords)                                                                           
        self.simulate(T,num_sweeps, relax=relax,monoclinic=monoclinic)

    def simulate(self, T, num_sweeps, relax=False, monoclinic=False):
        kT= T*8.6173303*10**(-5)
        for s in range(num_sweeps):
            for i in range(self.N):
                self.sweep(kT, volume_change=True)
                #self.energy_vals.append(self.energy)
            #self.optimize_struct(step=0.01, N_step=5, relax=False, fmax=0.5, relax_steps=2, monoclinic=monoclinic)


        cell = self.atoms.get_cell()

        self.optimize_struct(step=0.01, N_step=5, relax=relax, fmax=0.5, relax_steps=10, monoclinic=monoclinic)
        if relax:
            self.optimize_struct(step=0.01, N_step=5, relax=True, fmax=0.5, relax_steps=10, monoclinic=monoclinic)

        x_a = self.atoms.cell[0,0]/cell[0,0]
        x_c = self.atoms.cell[2,2]/cell[2,2]
        self.Va_pos[:,0:2] = x_a*self.Va_pos[:,0:2]
        self.Va_pos[:,2] = x_c*self.Va_pos[:,2]

        self.energy = self.atoms.get_potential_energy()
        if monoclinic:
            print(self.energy,self.atoms.cell[0,0]/self.x,np.linalg.norm(self.atoms.cell[1])/self.y,self.atoms.cell[2,2]/self.z)
        else:
            print(self.energy,self.atoms.cell[0,0]/self.x, self.atoms.cell[2,2]/self.z)
        #self.V = self.atoms.get_volume() - self.N_Li * (4/3) * math.pi * self.r_Li**3
        #self.pos = self.atoms.get_positions()
        self.energy_vals.append(self.energy)  



    def run_CMC_disorder(self, num_sweeps=1000, num_vacancies = 0, monoclinic=False):
        #kT= T*8.6173303*10**(-5) #eV
        #h = 4.135667696*10**(-15) #eV/s
        #mu_Li = -5.592988 #eV
        #Lambda = h /math.sqrt(2*math.pi*kT)
        #self.r_Li = 1.82 # 0.92 #angstrom
        sym = self.atoms.get_chemical_symbols()

        self.N = len([atom for atom in sym if atom=='Li']) #Number of MC spots
        self.initialize_vacancy(num_vacancies, relax=False, fmax=0.5)
        

        self.energy = self.atoms.get_potential_energy()
        self.energy_vals = [self.energy]

        self.pos = self.atoms.get_positions()


        self.N_Li = len(self.Li_coords)                                                                             
        self.N = 1 # to trick the simulate function
        for s in range(num_sweeps):
            self.swap()


        print(np.mean(self.energy_vals))
        

