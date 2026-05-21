1,2c1,2
< for(const auto& chunk : chunks){
<   std::vector<uint32_t>bytes;
---
> for (const auto& chunk : chunks) {
>   std::vector<uint32_t> bytes;
4,6c4,8
<   for(size_t i = 0 ; i < chunk.size() ; i++){
<     auto id = vocab.at(chunks[i]);
<     bytes.emplace_back(id);
---
> 
>   for (size_t i = 0; i < chunk.size(); i++) {
>     std::string byte_str(1, chunk[i]);
> 
>     bytes.emplace_back(vocab.at(byte_str)); 
8a11
> 
27c30,31
<     if (lowest_rank == SIZE_MAX) 
---
> 
>     if (lowest_rank == SIZE_MAX) {
28a33
>     }
34c39,40
<   for(const auto b : bytes){
---
> 
>   for (const auto b : bytes) {
38d43
< 
