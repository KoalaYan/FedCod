cd /dev/shm
fn=$(ls -l | grep yanpeishen | awk '{print $9}')
for i in $fn;do
rm $i
echo $i
done