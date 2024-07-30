import scipy as scp
import scipy.sparse as sps

# TODO: 
# * Add costs to measurements somehow
# * Maybe allow exporting to .pomdp files (https://www.pomdp.org/code/pomdp-file-spec.html)

class POMDP():
    
    def __init__(self, df=0.99):
        self.df = df

        # Currently only initializes variables
        self.P:dict
        self.Factors:dict; self.FactorNames:list
        self.Om_abs:dict
        self.Om_expl:dict
        self.Oa_expl:dict
        self.R:dict #technically a sparse matrix...
        self.ActionNames:dict
        self.S:int; self.A:int; self.M:dict[str,tuple[float,tuple[int,int]]]

    #############################################################
    #                   From PRISM to Python:
    #############################################################

    def import_POMDP_Prism(self, filename):
        """Imports entire model from Prism files."""
        self._import_states_Prism(filename+".sta")
        self._import_P_Prism(filename+".tra")
        self._import_R_Prism(filename+".trew")
        self._import_O_Prims(filename+".obs")


    def _import_states_Prism(self, filename):
        """Imports states from .sta prism file."""
        self.Factors = dict()
        with open(filename) as f:
            self.Factornames = f.readline()[1:-2].split(sep=",")
            for row in f:
                s, factors = row.split(sep=":")
                self.Factors[int(s)] = eval(factors)
        self.S = len(self.Factors)

    def _import_P_Prism(self, filename):
        """Imports transition function from .tra prism file."""
        
        self.P = dict()
        self.ActionNames = dict()

        with open(filename) as f:
            _header = f.readline()

            for row in f:
                row = row.split(sep=" ")
                l = len(row)
                s,a = row[0], row[l-1]
                a = a[:-1]
                if a in list(self.ActionNames.keys()):
                    a = self.ActionNames[a]
                else:
                    aname = a
                    a = int(len(self.ActionNames))
                    self.ActionNames[aname] = a
                    self.P[a] = sps.dok_array((self.S,self.S))
                for i in range(1,len(row)-1):
                    prob,snext = row[i].split(sep=":")
                    s,snext,prob = int(s), int(snext), float(prob)
                    self.P[a][s,snext] = prob
    
    def _import_R_Prism(self, filename):
        """Imports rewards from .trew prism file."""
        # Note: Prism assumes R is of form SxAxS. We'll assume it's SxA only, 
        # so we won't be needing all info from the file

        self.R = sps.dok_array((len(self.ActionNames), self.S))
        
        with open(filename) as f:
            _header = f.readline()

            for (i,row) in enumerate(f):
                a = i % len(self.ActionNames)
                row = row.split(sep=" ")
                s = int(row[0])
                if len(row) > 1:
                    r = float(row[1].split(sep=":")[0])
                    self.R[a,s] = r

    def _import_O_Prims(self, filename):
        """Imports observations from .obs prism file"""
        
    
    #############################################################
    #                   From .pomdp to Python:
    #############################################################

    # TBW
    #
    #
    #

    #############################################################
    #                   From Python to .(am)pomdp:
    #############################################################
    
    def export_pomdp(self, outfile, AM):

        # Open file, stop if one already exists
        try:
            f = open(outfile, "x")
            writing = True
        except FileExistsError:
            print("Warning: output file for explicit M already exists. File is not being saved.")
            return

        self.write_preamble(f, AM)
        self.write_transitions(f, AM)
        self.write_observations(f, AM)
        self.write_rewards(f,AM)

    def write_preamble(self, file, AM=False):
        actions, measurements, obs = "", "", ""
        for a in self.ActionNames.keys():
            actions+=str(a)+" "
        for m in self.Om_expl.keys():
            measurements+=str(m)+" "
        for o in self.M.keys():
            obs+=str(o)+" "

        if not AM:
            actions+=measurements
        
        file.write("""discount: {}\nvalues: reward\nstates: {}\nactions: {}\nobservations: {}""".format(self.df, self.S, actions, obs))
        
        if AM:
            file.write("""\nmeasurements: {}\n""".format(measurements))
    
    def write_transitions(self, file, AM=False):
        self.write_header(file, "Transitions")

        # Include measurement transitions if exported as .pomdp file
        if not AM:
            for o in self.Om_expl.keys():
                file.write("T: {} identity \n".format(o))
        
        for a in self.ActionNames.keys():
            anum = self.ActionNames[a]
            thisP = sps.coo_matrix(self.P[anum]) #removes zero elements
            for sfrom, sto, p in zip(thisP.row, thisP.col, thisP.data):
                file.write("T: {} : {} : {} {} \n".format(a,sfrom,sto,p))


    def write_observations(self, file, AM=False):
        self.write_header(file, "Observations")

        # Observations from actions
        if not AM and not self.Oa_expl:
            for a in self.ActionNames.keys():
                file.write("O : {} : * : NoObs 1.0 \n".format(a))
        elif not AM and self.Oa_expl:
            print("Warning: Oa not implemented yet!")

        # Observations form measurements
        for m in self.Om_expl.keys():
            thisO = sps.coo_matrix(self.Om_expl[m])
            for s, o, p in zip(thisO.row, thisO.col, thisO.data):
                file.write("O : {} : {} : {} {} \n".format(m,s,o,p))
        
    def write_rewards(self, file, AM=False):
        self.write_header(file, "Rewards")

        # Measuring Costs
        if not AM:
            for m in self.M.keys():
                file.write("R : {} : * : * : * : {} \n".format(m,self.M[m][0]))
        else:
            for m in self.M.keys():
                file.write("C : {} : {} \n".format(m, self.M[m][0]))

        # Rewards
        thisR = sps.coo_matrix(self.R)
        alist = list(self.ActionNames.keys())
        for (a,s,r) in zip(thisR.row, thisR.col, thisR.data):
            astr = alist[list(self.ActionNames.values()).index(a)]
            file.write("R : {} : {} : * : * : {} \n".format(astr,s,r)) # We assume R=AxS, pomdp allows AxSxSxO

    def write_header(self, file, txt):
        file.write("\n#################################################\n")
        file.write("#               {}        \n".format(txt))
        file.write("#################################################\n\n")
        


location= "Environments/Prismfiles/avoid/"; name="avoid"
M = AM_Model()
M.import_MDP_Prism(filename=location+name)
# M.import_states_Prism(location+"out.sta")
# # print(M.Factors)
# M.import_P_Prism(location+"out.tra")
# # for k in M.ActionNames.keys():
# #     i = M.ActionNames[k]
# #     print(k,i,M.P[i],"\n\n\n=====\n\n\n")
# print(M.ActionNames)
# M.import_M_abs(location+"out.amobs")
# # print(M.M_abs)
# M.build_explitit_M("out.amobsexpl")
# # print(M.M_expl)
# M.import_R_Prism(location+"out.trew")
M.export_pomdp(outfile=location+"out.ampomdp", AM=True)
