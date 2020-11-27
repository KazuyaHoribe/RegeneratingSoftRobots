<!-- 
<p align="center">
  <img src="images/carsmallest.gif" />
</p> -->
Regeneration
![](images/regeneration.gif)
Biped
![](images/biped.gif)
Tripod
![](images/tripod.gif)
Multiped
![](images/multiped.gif)
## How to run
<!-- <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Csqrt%7Bab%7D%20&bc=White&fc=Black&im=tif&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \sqrt{ab} " width="" height="" /> -->

```bash
# clone project   
git clone https://github.com/KazuyaHoribe/RegeneratingSoftRobots.git   

# install dependencies   
cd RegeneratingSoftRobots
pip install -r requirements.txt
```

You can run GA using a like below command.
```
python main_creatures.py --im_size 7 --number_neighbors 7 --popsize 50 --generations 101 --sigma 0.05 --N 10 --threads 4 --fig_output_rate 10
```

When you see a locomotion of a virtual creature, you need to build a physical simulator named "VoxCad" following this direction.
https://github.com/skriegman/evosoro  
NOTE: This simulator works only on Linux.