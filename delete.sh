cd /dev/shm
fn=$(ls -l | grep ubuntu | awk '{print $9}')
for i in $fn;do
rm $i
echo $i
done