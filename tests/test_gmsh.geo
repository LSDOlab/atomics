// Gmsh project created on Mon Sep 14 10:49:35 2020
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {20, 0, 0, 1.0};
//+
Point(3) = {20, 10, 0, 1.0};
//+
Point(4) = {10, 10, 0, 1.0};
//+
Point(5) = {10, 30, 0, 1.0};
//+
Point(6) = {0, 30, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {3, 4};
//+
Line(3) = {5, 5};
//+
Line(4) = {5, 6};
//+
Line(5) = {1, 1};
//+
Line(6) = {2, 3};
//+
Line(7) = {4, 5};
//+
Line(8) = {6, 1};
//+
Curve Loop(1) = {7, 4, 8, 1, 6, 2};
//+
Plane Surface(1) = {1};
//+
Field[1] = AutomaticMeshSizeField;
//+
Characteristic Length {6, 5, 4, 3, 2, 1} = 0.1;
//+
Field[1].NRefine = 10;
//+
Field[1].NRefine = 5;
//+
Delete Field [1];
