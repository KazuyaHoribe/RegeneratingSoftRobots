<!-- 
<p align="center">
  <img src="images/carsmallest.gif" />
</p> -->
# Regenerating soft robots through neural cellular automata  

### Regeneration  
![](images/regeneration.gif)  
### Biped  
![](images/biped.gif)  
### Tripod  
![](images/tripod.gif)  
### Multiped  
![](images/multiped.gif)  
## How to run
<!-- <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Csqrt%7Bab%7D%20&bc=White&fc=Black&im=tif&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \sqrt{ab} " width="" height="" /> -->

Firstly, install Anaconda as python 2.7 distribution on your linux machine.  

```bash
# clone project   
git clone https://github.com/KazuyaHoribe/RegeneratingSoftRobots.git   

# install dependencies    
cd RegeneratingSoftRobots 
pip install -r requirements.txt
```

You can run GA using a following command.
```
python main_creatures.py --number_neighbors 7 --popsize 50 --generations 101 --sigma 0.03 --N 10 --threads 1 --fig_output_rate 10
```

When you see a locomotion of a virtual creature, you need to build a physical simulator named "VoxCad" following this direction.
https://github.com/skriegman/evosoro  

After evolving soft robots, you can test their regeneration using a following command.

```
python regeneration_task.py --number_neighbors 7 --popsize 50 --generations 101 --sigma 0.03 --N 10 --threads 1 --fig_output_rate 10
```