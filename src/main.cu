#include <cstdint>
#include <cstdio>
#include <cstring>

#include "io.hpp"
#include "stages.hpp"

using namespace std;

int main(int argc, char *argv[])
{
  setbuf(stdout, NULL);

  init_libff();

  bool is_stage_0 = strcmp(argv[1], "compute-stage-0") == 0;
  bool is_stage_1 = strcmp(argv[1], "compute-stage-1") == 0;
  bool is_stage_2 = strcmp(argv[1], "compute-stage-2") == 0;
  bool is_stage_3 = strcmp(argv[1], "compute-stage-3") == 0 || strcmp(argv[1], "compute") == 0;
  bool is_stage_4 = strcmp(argv[1], "MNT4753") == 0 || strcmp(argv[1], "MNT6753") == 0;

  if (is_stage_4)
  {
    const char *curve = argv[1];
    const char *mode = argv[2];
    const char *params_path = argv[3];
    const char *input_path = argv[4];
    const char *output_path = argv[5];

    FILE *input_params = fopen(params_path, "r");
    FILE *inputs = fopen(input_path, "r");
    FILE *outputs = fopen(output_path, "w");

    stage_4(curve, mode, input_params, inputs, outputs);
  }
  else
  {

    FILE *inputs = fopen(argv[2], "r");
    FILE *outputs = fopen(argv[3], "w");

    if (is_stage_0)
    {
      //removed: stage_0(inputs, outputs);
    }
    else if (is_stage_1)
    {
      //removed: stage_1(inputs, outputs);
    }
    else if (is_stage_2)
    {
      //removed: stage_2(inputs, outputs);
    }
    else if (is_stage_3)
    {
      //removed: stage_3(inputs, outputs);
    }
  }
  return 0;
}
