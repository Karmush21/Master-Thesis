
# Special patients: 2 - only 7 timepoints, 5, missing one timepoint
# Patietns to remove: 8, 11, 48, 59: Maybe we fix these later, maybe not.
'''
Notes:
It's quite clear already from the first time point which patients are higher quality,
maybe something to consider when you're doing the split later.

Maybe look again at the patients that have similar issues as pt59, maybe some are worth
fixing.

It would be stupid trying to fix all because the boundaries get quite roughly shape,
and we still want the network to learn these.

Perhaps try a smooting again and see if that helps anything creating a smoother
surface? However you would have to do this for all of them... which might not
be want you want. 

THE SCALE 3D TOOL IS REALLY GOOD FOR REMOVING THE OVERSEGMENTATION FROM LEES_ALG!!!!!

'''

data = {
    
    "pt3": 
{
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6",
            "dt7", "dt8", "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15",
            "dt16", "dt17", "dt18", "dt19", "dt20"], 

    "lees_alg": []

},


#Commnets
"pt4": 
{
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", 
                "dt6", "dt7", "dt8", "dt9", "dt10", "dt11", "dt12", "dt13", ], 

    "lees_alg": ["dt14", "dt15", "dt16", "dt17", "dt18", "dt19", "dt20"] 
},


#Comments, many have the oversegmetnation now at the valve plane.
"pt5": 
{
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7",
    "dt8", "dt9", "dt10", "dt11", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},


"pt6": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},

#good quality image
"pt7": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt10", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17", "dt20"], 

    "lees_alg": ["dt9", "dt11", "dt18", "dt19"]
},

#We're skipping 8 for now as that one had some issues, we excluded that one before as well
"pt8": 
{   
    "anmars_alg": [], 

    "lees_alg": []
},


"pt10": 
{   
    "anmars_alg": ["dt2", "dt4", "dt5", "pt6", "dt7", "dt8", "dt9", "dt10", "dt13", "dt14",
    "dt15", "dt16", "dt17", "dt18"], 

    "lees_alg": ["dt1", "dt3", "dt11", "dt12", "dt19", "dt20" ]
},


#We're skipping 11 for now as that also had some issues, we exluced that one before as well
"pt11": 
{   
    "anmars_alg": [], 

    "lees_alg": []
},

"pt12": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},


"pt13": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},

"pt14": 
{   
    "anmars_alg": ["dt3", "dt4", "dt7", "dt8", "dt9", "dt10", "dt11"], 

    "lees_alg": ["dt1", "dt2", "dt5", "dt6", "dt12", "dt13", "dt14", "dt15", "dt16",
    "dt17", "dt18", "dt19", "dt20"]
},

#I mean there's very slighly oversegmentation compared to the others pateitns. however mine
#produces like 4 holes for the early volumes which takes a lot of time to fill in. 

#TODO put this one in validation I think. 
"pt15": 
{   
    "anmars_alg": ["dt1", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8", "dt9"], 

    "lees_alg": ["dt2", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16",
    "dt17", "dt18", "dt19", "dt20"]
},

# Good quality patient
"pt16": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16",
    "dt17", "dt18", "dt19", "dt20"], 

    "lees_alg": []
},

# Low quality patient
"pt17": 
{   
    "anmars_alg": ["dt2", "dt3", "dt5", "dt6", "dt7", "dt8", "dt9",
    "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt19", "dt20"], 

    "lees_alg": ["dt1", "dt4", "dt10", "dt17", "dt18"]
},

# dt12 and a couple of other timepoints has similar overseg has pt59, good quality patient otherwise
"pt18": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8", "dt9",
    "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17", "dt18",
    "dt19", "dt20"], 

    "lees_alg": []
},

# Has similar overseg issues as pt59, at the bottom, can probably remove these one day
"pt19": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt20"], 

    "lees_alg": ["dt19"]
},

# A lot of oversg as pt59, I think I'll fix all of them.
# TODO Try to low postporcessing on this volume, like smooting and filling of holes.
"pt20": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13" ], #TODO fix the last one later please 

    "lees_alg": []
},

"pt21": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19"], 

    "lees_alg": ["dt4", "dt11", "dt20"]
},

# Good quality patient
"pt22": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},


# Had to fix a lot of these.
"pt23": 
{   
    "anmars_alg": ["dt1", "dt2", "dt4", "dt5", "dt6", "dt7", "dt8","dt9",
    "dt10", "dt14", "dt20"], 

    "lees_alg": ["dt3", "dt11", "dt12", "dt13", "dt15", "dt16", "dt17", 
    "dt18", "dt19"]
},

# a lot of so clear oversegmetnaton. Maybe fix this one eventually.
"pt24": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt14", "dt15", "dt16", "dt17", "dt18",
    "dt19", "dt20"], 

    "lees_alg": ["dt13"]
},


# Contain some clear oversegmentation that perhaps is worth it to fix. e.g. dt8
"pt25": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt19",
    "dt20"], 

    "lees_alg": ["dt17", "dt18"]
},


# This is the one that has some weird artifact on it. A lot of volumes here
# that can easily be fixed.
"pt26": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt9", "dt10", "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"], 

    "lees_alg": []
},

#This had one had some issues, maybe better to fix lee's alg instead of mine. mine has huge holes!
# TODO Maybe we chill witht his one for now.
"pt27": 
{   
    "anmars_alg": [], 

    "lees_alg": []
},


# This one has two clear oversegmentations. At valve plane and at the bottom. I've only
# remove the one at the valve plane for now. To be consistent. 
# This is one that has ct ARTIFACT
"pt28": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8",
    "dt11", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17"], 

    "lees_alg": ["dt9", "dt10", "dt18", "dt19", "dt20"]
},

# This is one that has CT ARTIFACT
"pt29": 
{   
    "anmars_alg": ["dt3", "dt4", "dt5", "dt6", "dt7", "dt8", "dt9",
    "dt10", "dt11"], 

    "lees_alg": ["dt1", "dt2", "dt12", "dt13", "dt14", "dt15", "dt16", "dt17",
    "dt18", "dt19", "dt20"]
},

"pt30": 
{   
    "anmars_alg": ["dt1", "dt2", "dt3"], 

    "lees_alg": []
}




#TODO if len(anmars_alg) + len(lees_alg) != 20 : quit





}


print(len(data.keys()))
