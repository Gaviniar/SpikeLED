#include <torch/extension.h>
#include <random>
#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

#define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
    [&] {                                          \
        if (optional_value.has_value())            \
        {                                          \
            const bool HAS_VALUE = true;           \
            return __VA_ARGS__();                  \
        }                                          \
        else                                       \
        {                                          \
            const bool HAS_VALUE = false;          \
            return __VA_ARGS__();                  \
        }                                          \
    }()

torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace);

// Returns sampled neighbor indices
torch::Tensor sample_neighber_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace)
{
    CHECK_CPU(rowptr);
    CHECK_CPU(col);
    CHECK_CPU(idx);
    CHECK_INPUT(idx.dim() == 1);

    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    std::vector<int64_t> n_ids;
    std::random_device rd;
    std::mt19937 gen(rd());

    int64_t n, c, e, row_start, row_end, row_count;

    if (num_neighbors < 0)
    { // No sampling ======================================
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            for (int64_t j = 0; j < row_count; j++)
            {
                e = row_start + j;
                c = col_data[e];
                n_ids.push_back(c);
            }
        }
    }
    else if (replace)
    { // Sample with replacement (真正的有放回采样) ===============================
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            if (row_count == 0) continue; // Skip nodes with no neighbors

            std::uniform_int_distribution<int64_t> dis(0, row_count - 1);
            
            for (int64_t j = 0; j < num_neighbors; j++)
            {
                int64_t rand_idx = dis(gen);
                e = row_start + rand_idx;
                c = col_data[e];
                n_ids.push_back(c);
            }
        }
    }
    else
    { // Sample without replacement via Robert Floyd algorithm ============
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            std::unordered_set<int64_t> perm;
            if (row_count <= num_neighbors)
            {
                for (int64_t j = 0; j < row_count; j++)
                    perm.insert(j);
            }
            else
            { // Robert Floyd algorithm
                std::uniform_int_distribution<int64_t> dis(0, row_count - 1);
                for (int64_t j = row_count - num_neighbors; j < row_count; j++)
                {
                    int64_t rand_val = std::uniform_int_distribution<int64_t>(0, j)(gen);
                    if (!perm.insert(rand_val).second)
                        perm.insert(j);
                }
            }

            for (const int64_t &p : perm)
            {
                e = row_start + p;
                c = col_data[e];
                n_ids.push_back(c);
            }
        }
    }

    int64_t N = n_ids.size();
    auto out_n_id = torch.from_blob(n_ids.data(), {N}, col.options()).clone();
    return out_n_id;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sample_neighber_cpu", &sample_neighber_cpu, "Node neighborhood sampler");
}
