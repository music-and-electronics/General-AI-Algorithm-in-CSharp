using System;
using System.Collections.Generic;
using System.Linq;

namespace Ai_hw_8
{
    class Layer
    {
        public int neurons;
        public int inputs;
        public List<float[]> weights = new List<float[]>();
        public List<float> values = new List<float>();
        public List<float> original = new List<float>();
       
        public void Result_Values(List<float>pre_values, List<float[]>weights)
        {
            foreach (float[] a in weights)
            {
                values.Add(Activation_RELU(Multi_Weights(pre_values,a)));
            }
           
        }

        float Activation_RELU(float weight)
        {
            float tmp_weight;
            tmp_weight = weight;
            if (weight < 0)
            {
                return weight = (float)0.01 * tmp_weight;
            }
            else
            {
                return weight = tmp_weight;
            }
        }

        public float Multi_Weights(List<float> a, float[] b)
        {
            List<float> sum = new List<float>();
            if (a.Count > b.Length)
            {
                for (int i = 0; i < b.Length; i++)
                {
                    float tmp_sum = a[i] * b[i];
                    sum.Add(tmp_sum);
                }
            }
            else
            {
                for (int i = 0; i < a.Count; i++)
                {
                    float tmp_sum = a[i] * b[i];
                    sum.Add(tmp_sum);
                }
            }
            return sum.Sum();
        }
    }

    class Initial_DATA
    {
        public int layer_num;
        public Layer[] layers;

        public Initial_DATA(int layer_num)
        {
            this.layer_num = layer_num;
            layers = new Layer[layer_num];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer();
            }

            for (int i = 0; i < layer_num; i++)
            {
                if (i == 0)
                {
                    Console.Write("Input the number of Inputs: ");
                    layers[i].inputs = Int32.Parse(Console.ReadLine());
                    layers[i].neurons = layers[i].inputs;
                    Console.WriteLine("-Input the input values-");
                    for (int j = 0; j < layers[i].inputs; j++)
                    {
                        Console.Write($"{j} ");
                        layers[0].values.Add(float.Parse(Console.ReadLine()));
                    }
                }

                else if (i == layer_num - 1)
                {
                    Console.Write("Input the output number: ");
                    layers[i].neurons = Int32.Parse(Console.ReadLine());

                    for (int j = 0; j < layers[i].neurons; j++)
                    {
                        float[] weight = new float[layers[i-1].neurons];
                        layers[i].weights.Add(weight);
                        Console.Write("Input the original value: ");
                        layers[i].original.Add(float.Parse(Console.ReadLine()));
                    }

                    foreach (float[] a in layers[i].weights)
                    {
                        for (int k = 0; k < a.Length; k++)
                        {
                            a[k] = 1;
                        }
                    }
                    layers[i].Result_Values(layers[i - 1].values, layers[i].weights);
                }

                else
                {
                    layers[i].neurons = 2;
                    if (i - 1 == 0)
                    {
                        for (int j = 0; j < layers[i].neurons; j++)
                        {
                            float[] weight = new float[layers[i-1].inputs];
                            layers[i].weights.Add(weight);
                        }
                    }

                    else
                    {
                        for (int j = 0; j < layers[i].neurons; j++)
                        {
                            float[] weight = new float[layers[i-1].neurons];
                            layers[i].weights.Add(weight);
                        }
                    }

                    foreach (float[] a in layers[i].weights)
                    {
                        for (int k = 0; k < a.Length; k++)
                        {
                            a[k] =1;
                        }
                    }

                    layers[i].Result_Values(layers[i - 1].values, layers[i].weights);
                }
            }
        }
    }

    class ANN
    {

        public ANN()
        {
            Console.Write("Input the level of hidden layer: ");
            int layer_num = Int32.Parse(Console.ReadLine())+2;
            Initial_DATA initial = new Initial_DATA(layer_num);

            for (int num = 0; (Math.Abs((double)(initial.layers[layer_num - 1].original[0] - initial.layers[layer_num - 1].values[initial.layers[layer_num - 1].values.Count - 1])) > 10); num++)
            {
                Back_Propagation(ref initial);

                for (int i = 1; i < layer_num; i++)
                {
                    initial.layers[i].Result_Values(initial.layers[i].original, initial.layers[i].weights);
                }

                for(int i=1;i<layer_num;i++)
                {
                    for (int j = 0; j < initial.layers[i].weights.Count; j++)
                    {
                        for (int k = 0; k < initial.layers[i].weights[j].Length; k++)
                        {
                            Console.Write($"a[{j}][{k}]={initial.layers[i].weights[j][k]} ");
                        }
                        Console.WriteLine();
                    }
                }

            }


        }
        void Back_Propagation(ref Initial_DATA list)
        {
            int size = list.layer_num;
            List<float> original_tmp = new List<float>();
            for (int i = size - 1; i > 0; i--)
            {

                if (i == size - 1)
                {
                    float[] sub_val = new float[list.layers[i].neurons];
                    for (int j = 0; j < list.layers[i].neurons; j++)
                    {

                        sub_val[j] = list.layers[i].original[j] - list.layers[i].values[j];
                        for (int k = 0; k < list.layers[i].weights[j].Length; k++)
                        {
                            if (sub_val[j] != 0)
                                list.layers[i].weights[j][k] += list.layers[i].weights[j][k] / sub_val[j];
                        }
                    }
                }

                else
                {
                    float[] sub_val = new float[list.layers[i].neurons];
                    for (int j = 0; j < list.layers[i].neurons; j++)
                    {
                        List<float> original = new List<float>();
                        for (int k = 0; k < list.layers[i + 1].original.Count; k++)
                        {
                                original.Add(list.layers[i + 1].original[k] / list.layers[i + 1].weights[j][k]);
                        }

                        list.layers[i].original.Add(original.Sum());

                    }

                    for (int j = 0; j < list.layers[i].neurons; j++)
                    {
                        sub_val[j] = list.layers[i].original[j] - list.layers[i].values[j];
                        for (int k = 0; k < list.layers[i].weights[j].Length; k++)
                        {
                            if (sub_val[j] != 0)
                                list.layers[i].weights[j][k] += list.layers[i].weights[j][k] / sub_val[j];
                        }

                    }
                }
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            ANN ann = new ANN();
        }
    }
}
