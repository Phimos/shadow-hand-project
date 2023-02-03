



```
cd multical-input
```

intrinsic & extrinsic 
```
multical calibrate --boards ../charuco.yam
```

intrinsic only
```
multical intrinsic --boards ../charuco.yam
```

extrinsic only
```
multical calibrate --boards ../charuco.yaml --calibration intrinsic.json --fix_intrinsic
```

```
multical calibrate --boards ../charuco.yaml --calibration calibration.json --fix_intrinsic --fix_camera_poses
```


```
Number of cameras found: 4
Width: 1280, Height: 720
------------------------------
Index: 0
Serial number: 125322061389
Intrinsics:
        fx:  912.2255859375
        fy:  911.4783325195312
        ppx: 645.8405151367188
        ppy: 349.43603515625
------------------------------
Index: 1
Serial number: 125322060991
Intrinsics:
        fx:  911.5418090820312
        fy:  911.1187744140625
        ppx: 621.3646240234375
        ppy: 354.3831787109375
------------------------------
Index: 2
Serial number: 125322064398
Intrinsics:
        fx:  915.0556030273438
        fy:  914.1288452148438
        ppx: 641.2314453125
        ppy: 363.4847412109375
------------------------------
Index: 3
Serial number: 125322061981
Intrinsics:
        fx:  913.5755004882812
        fy:  912.9376831054688
        ppx: 628.3202514648438
        ppy: 360.56365966796875
```