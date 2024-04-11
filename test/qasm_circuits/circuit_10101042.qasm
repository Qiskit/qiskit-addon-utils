OPENQASM 3.0;
include "stdgates.inc";
gate sxdg _gate_q_0 {
  s _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
}
gate xx_minus_yy_5477468176(_gate_p_0, _gate_p_1) _gate_q_0, _gate_q_1 {
  rz(-3.3600737387696333) _gate_q_1;
  rz(-pi/2) _gate_q_0;
  sx _gate_q_0;
  rz(pi/2) _gate_q_0;
  s _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  ry(1.836592803320145) _gate_q_0;
  ry(-1.836592803320145) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  sdg _gate_q_1;
  rz(-pi/2) _gate_q_0;
  sxdg _gate_q_0;
  rz(pi/2) _gate_q_0;
  rz(3.3600737387696333) _gate_q_1;
}
gate rzx_4584323792(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
}
gate rzx_4584324112(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
}
gate ecr _gate_q_0, _gate_q_1 {
  rzx_4584323792(pi/4) _gate_q_0, _gate_q_1;
  x _gate_q_0;
  rzx_4584324112(-pi/4) _gate_q_0, _gate_q_1;
}
gate cs _gate_q_0, _gate_q_1 {
  p(pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  p(-pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  p(pi/4) _gate_q_1;
}
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
gate rzz_5458600272(_gate_p_0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(5.093943105338468) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
gate rxx_5485069456(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(0.47290716693712) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
gate rxx_5485068688(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(1.3043889125856758) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
gate rcccx _gate_q_0, _gate_q_1, _gate_q_2, _gate_q_3 {
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  cx _gate_q_0, _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_1, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
  u1(pi/4) _gate_q_3;
  cx _gate_q_2, _gate_q_3;
  u1(-pi/4) _gate_q_3;
  u2(0, pi) _gate_q_3;
}
gate iswap _gate_q_0, _gate_q_1 {
  s _gate_q_0;
  s _gate_q_1;
  h _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  cx _gate_q_1, _gate_q_0;
  h _gate_q_1;
}
gate ryy_5492943056(_gate_p_0) _gate_q_0, _gate_q_1 {
  rx(pi/2) _gate_q_0;
  rx(pi/2) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(1.9651300768398978) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rx(-pi/2) _gate_q_0;
  rx(-pi/2) _gate_q_1;
}
gate rzz_5492933520(_gate_p_0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(3.544083053395088) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
gate rxx_5492947472(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(5.0360498515356875) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
gate csdg _gate_q_0, _gate_q_1 {
  p(-pi/4) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  p(pi/4) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  p(-pi/4) _gate_q_1;
}
gate rccx _gate_q_0, _gate_q_1, _gate_q_2 {
  u2(0, pi) _gate_q_2;
  u1(pi/4) _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  u1(-pi/4) _gate_q_2;
  cx _gate_q_0, _gate_q_2;
  u1(pi/4) _gate_q_2;
  cx _gate_q_1, _gate_q_2;
  u1(-pi/4) _gate_q_2;
  u2(0, pi) _gate_q_2;
}
gate rxx_5492945552(_gate_p_0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(5.314238241279584) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
gate xx_plus_yy_5492938768(_gate_p_0, _gate_p_1) _gate_q_0, _gate_q_1 {
  rz(5.99990592172356) _gate_q_0;
  rz(-pi/2) _gate_q_1;
  sx _gate_q_1;
  rz(pi/2) _gate_q_1;
  s _gate_q_0;
  cx _gate_q_1, _gate_q_0;
  ry(-0.1370257507100862) _gate_q_1;
  ry(-0.1370257507100862) _gate_q_0;
  cx _gate_q_1, _gate_q_0;
  sdg _gate_q_0;
  rz(-pi/2) _gate_q_1;
  sxdg _gate_q_1;
  rz(pi/2) _gate_q_1;
  rz(-5.99990592172356) _gate_q_0;
}
qubit[10] q;
xx_minus_yy_5477468176(3.67318560664029, 3.3600737387696333) q[9], q[8];
u3(3.7855699412285815, 1.6597892246446815, 0.09214153194304223) q[4];
id q[5];
y q[1];
p(1.4555562415381211) q[0];
cu(5.5881071713373744, 5.627478641101485, 0.430687555567003, 2.5386991013207507) q[2], q[6];
cp(5.3367323551704935) q[3], q[7];
cry(4.268445006731576) q[7], q[6];
crz(1.674359275470625) q[4], q[9];
h q[5];
cu(3.177905164772013, 2.0621320339362565, 4.310195961254752, 4.5336517936286755) q[8], q[0];
ecr q[3], q[2];
y q[1];
cs q[4], q[7];
ccz q[3], q[6], q[0];
cs q[8], q[5];
id q[9];
rzz_5458600272(5.093943105338468) q[2], q[1];
p(1.2360822329082872) q[7];
rxx_5485069456(0.47290716693712) q[1], q[5];
rxx_5485068688(1.3043889125856758) q[2], q[4];
cu(0.0708867113663714, 2.3851497152821537, 4.708767253947128, 4.122429156551058) q[3], q[8];
x q[0];
rx(0.2322336776732453) q[6];
y q[9];
id q[3];
h q[0];
ccx q[6], q[7], q[5];
sxdg q[1];
rcccx q[9], q[8], q[2], q[4];
iswap q[2], q[0];
sx q[9];
sx q[6];
rx(3.941441329429165) q[3];
p(2.6977122890836003) q[1];
cp(6.02843564793497) q[7], q[5];
t q[8];
sdg q[4];
sxdg q[8];
cswap q[6], q[4], q[7];
ryy_5492943056(1.9651300768398978) q[3], q[0];
s q[1];
cry(5.227584781453966) q[5], q[9];
id q[2];
cswap q[2], q[3], q[4];
x q[0];
rzz_5492933520(3.544083053395088) q[1], q[7];
u1(4.621042063722644) q[6];
id q[5];
swap q[9], q[8];
rxx_5492947472(5.0360498515356875) q[2], q[1];
U(1.4940425068787202, 4.485777116417918, 2.8349617570528896) q[9];
csdg q[8], q[0];
cy q[5], q[3];
rccx q[6], q[7], q[4];
ch q[9], q[2];
rxx_5492945552(5.314238241279584) q[7], q[8];
sdg q[1];
sxdg q[4];
xx_plus_yy_5492938768(0.2740515014201724, 5.99990592172356) q[6], q[3];
rx(0.8052231441649303) q[0];
t q[5];
