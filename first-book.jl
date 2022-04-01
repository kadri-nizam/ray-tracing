### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ ef4c7762-b0c3-11ec-1425-cd672582f162
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 2cb0d923-bd4c-4216-b000-70873e41b8e8
begin
	using Plots
	using Images
	using LinearAlgebra
	using Random
	using Distributions
end

# ╔═╡ 661dae0b-fab4-4c17-a7ed-de4439aaf132
begin
	const ASPECT_RATIO = 16/9
	
	const IMAGE_WIDTH = 200
	const IMAGE_HEIGHT = Int(IMAGE_WIDTH ÷ ASPECT_RATIO)
	
	const X = 1
	const Y = 2
	const Z = 3
end;

# ╔═╡ 8cc10884-ad70-412c-89c0-f4957016a045
begin
	struct Color{T<:Real}
		red::T
		green::T
		blue::T
		
		function Color{T}(red::Real, green::Real, blue::Real) where T<:Real			
			new(red, green, blue)
		end
		
		function Color{T}(color_array::Vector) where T<:Real
			red, green, blue = color_array
			Color{T}(red, green, blue)
		end
	end

	# Overator overloading
	function Base.zero(::Color{T}) where {T}
		Color{T}(0, 0, 0)
	end
	
	function Base.:+(c1::Color{T}, c2::Color{T}) where {T}
		return Color{T}(
			c1.red + c2.red,
			c1.green + c2.green,
			c1.blue + c2.blue
		)
	end

	function Base.:+(c1::Color{T}, c2::Vector) where {T}
		return Color{T}(
			c1.red + c2[X],
			c1.green + c2[Y],
			c1.blue + c2[Z]
		)
	end
	
	function Base.:*(λ::Number, c::Color{T},) where {T}
		return Color{T}(
			λ * c.red, 
			λ * c.green, 
			λ * c.blue
		)
	end
end

# ╔═╡ e54406fe-0be0-493a-bbd5-98e9f3146a8f
struct Ray
	origin::Vector{Float64}
	direction::Vector{Float64}
end

# ╔═╡ 34cb88ac-b5d1-4309-9edd-65e7127094ee
begin
	const ORIGIN = [0; 0; 0]

	const VIEWPORT_HEIGHT = 2
	const VIEWPORT_WIDTH = ASPECT_RATIO * VIEWPORT_HEIGHT
	const FOCAL_LENGTH = 1.0
	
	const HORIZONTAL = [VIEWPORT_WIDTH; 0; 0]
	const VERTICAL = [0; VIEWPORT_HEIGHT; 0]
	
	const TOP_LEFT_CORNER = (
		ORIGIN - 0.5HORIZONTAL - 0.5VERTICAL - [0; 0; FOCAL_LENGTH]
	)

	const SAMPLES_PER_PIXEL = 10
	const MAX_DEPTH = 5
end;

# ╔═╡ 6e0d0040-b8ee-459e-b694-811ce03dd961


# ╔═╡ 30549a0b-bf46-4326-a49e-3ec704768f0b
mutable struct HitRecord
	t::Real
	location::Vector
	normal::Vector
	front_face::Bool

	function HitRecord()
		new(
			0,
			[0; 0; 0],
			[0; 0; 0],
			false
		)
	end

end

# ╔═╡ 3e0c6fd2-0a60-4233-8cac-8cab8cc31466
abstract type AbstractObject end

# ╔═╡ dbf8802b-1b70-424d-bd50-9f87d259db5c
struct Sphere <: AbstractObject
	center::Vector
	radius::Real	
end

# ╔═╡ 4c1b0089-c914-4321-a63b-822373be067d
function set_face_normal(ray::Ray, out_normal::Vector)
	front_face = (ray.direction' * out_normal) < 0
	normal =  front_face ? out_normal : -out_normal

	return normal, front_face
end

# ╔═╡ 71b959f9-9fc4-4e22-9f64-623ada64d111
md"""
# Working Helper Functions
"""

# ╔═╡ 1b0094ce-71b8-4bb9-a2e1-53d716b70880
function random_in_unit_sphere()
	while true
		v = rand(Uniform(-1, 1), 3)

		if norm(v) <= 1
			return v
		end
	end
end
		

# ╔═╡ 517ff8a4-a3fc-43c9-adcc-4fcfc2eb449f
function hit(center::Vector, radius::Real, ray::Ray)
	oc = ray.origin - center
	a = ray.direction' * ray.direction
	half_b = oc' * ray.direction
	c = oc'oc - radius^2

	discriminant = half_b^2 - a*c
	
	if discriminant < 0
		return -1
	else
		return -(half_b+sqrt(discriminant))/a
	end
end

# ╔═╡ a8d48d9d-821c-47f4-aaf2-931086cbfd1e
function row_ratio(row::Real)
	return (row - 1)/(IMAGE_HEIGHT - 1)
end

# ╔═╡ c4940508-567f-494a-8612-a226236ac15d
function column_ratio(column::Real)
	return (column - 1)/(IMAGE_WIDTH - 1)
end

# ╔═╡ 44f9dd29-6ced-4f79-925b-8c1d5eda8ac7
function unit_vector(x::Vector)
	return x/norm(x)
end

# ╔═╡ 86a6e1c7-c8f6-4a2e-ab1b-ce6c7b1684c6
function ray_at(ray::Ray, t::Float64)
	return ray.origin + t*ray.direction
end

# ╔═╡ 5257c0a3-7cd6-4b4c-ac8b-123116c9f963
function hit!(obj::Sphere, ray::Ray, tlim::NamedTuple, record::HitRecord)
	oc = ray.origin - obj.center
	a = ray.direction' * ray.direction
	half_b = oc' * ray.direction
	c = oc' * oc - obj.radius^2

	discriminant = half_b^2 - a*c
	
	if discriminant < 0
		return false
	end

	sqrtd = sqrt(discriminant)
	root = -(half_b + sqrtd) / a
	
	if root < tlim.min || tlim.max < root
		
		root = (-half_b + sqrtd) / a
		
		if root < tlim.min || tlim.max < root
			return false
		end
		
	end

	location = ray_at(ray, root)
	normal, front_face = set_face_normal(
		ray, 
		(location - obj.center)/obj.radius
	)

	record.t = root
	record.location = location
	record.normal = normal
	record.front_face = front_face
	
	return true
		
end

# ╔═╡ 3a72f057-a8c2-4531-a2e6-c438157105d8
function object_hit!(
	obj::Sphere, 
	ray::Ray, 
	tlim::NamedTuple,
	record::HitRecord
)

	hit_anything = hit!(obj, ray, tlim, record)

	return hit_anything
	
end

# ╔═╡ 97f17bbb-fa79-48cd-91a7-be72dca5820b
function object_hit!(
	obj_list::Vector{Sphere}, 
	ray::Ray, 
	tlim::NamedTuple,
	record::HitRecord
)
	hit_anything = false
	closest_so_far = tlim.max

	for obj in obj_list
		
		if hit!(obj, ray, (min = tlim.min, max = closest_so_far), record)
			hit_anything = true
			closest_so_far = record.t
		end
		
	end

	return hit_anything
end

# ╔═╡ 1c72807c-a8e5-4416-b8e4-80b23d0f36ea
function ray_color(ray::Ray, obj::Sphere)

	record = HitRecord()

	if object_hit!(obj, ray, (min = 0, max = Inf), record)
		return 0.5(Color{Float64}(1, 1, 1) + record.normal)
	end

	unit_direction = unit_vector(ray.direction)
	t = 0.5unit_direction[Y] + 1
	
	return (1 - t)*Color{Float64}(1.0, 1.0, 1.0) + t*Color{Float64}(0.5, 0.7, 1.0)
	
end

# ╔═╡ cb6d5996-49c1-4d79-99d3-0713bbfaefd3
function ray_color(ray::Ray, obj_list::Vector{Sphere}, depth::Int)

	if depth <= 0
		return Color{Float64}(0, 0, 0)
	end
	
	record = HitRecord()

	if object_hit!(obj_list, ray, (min = 0, max = Inf), record)
		target = record.location + record.normal + random_in_unit_sphere()
		return 0.5ray_color(
			Ray(record.location, target - record.location), 
			obj_list,
			depth-1
		)
	end

	unit_direction = unit_vector(ray.direction)
	t = 0.5unit_direction[Y] + 1
	
	return (1 - t)*Color{Float64}(1.0, 1.0, 1.0) + t*Color{Float64}(0.5, 0.7, 1.0)
	
end

# ╔═╡ 0246705a-2f60-4ac9-8ba1-7cbba560a442
function ray_color(ray::Ray)

	t = hit([0; 0; -1], 0.5, ray)
	
	if t > 0
		N = unit_vector(ray_at(ray, t) - [0; 0; -1]) .+ 1
		return 0.5Color{Float64}(N)
	end
	
	unit_direction = unit_vector(ray.direction)
	t = 0.5unit_direction[Y] + 1
	
	return (1 - t)*Color{Float64}(1.0, 1.0, 1.0) + t*Color{Float64}(0.5, 0.7, 1.0)
end

# ╔═╡ 23244102-6c96-4d60-bb14-236b54e384d8
function find_pixel(u, v)
	return TOP_LEFT_CORNER + u*HORIZONTAL + v*VERTICAL - ORIGIN
end

# ╔═╡ 39df6e80-0779-4937-b62f-d13e7b4a948d
function color_as_matrix(img::Matrix{Color})

	c_mat = zeros(3, IMAGE_HEIGHT, IMAGE_WIDTH)
	
	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT
			c_mat[X, row, column] = img[row, column].red
			c_mat[Y, row, column] = img[row, column].green
			c_mat[Z, row, column] = img[row, column].blue
		end
	end
	
	return c_mat
	
end

# ╔═╡ 624d5e30-2558-4aa5-9820-76f6556984a5
function imshow(color_matrix::Array{Float64, 3})
	colorview(RGB, color_matrix[:, end:-1:1, :])
end

# ╔═╡ 518200c4-dffa-44a5-ac3c-3dd1d1564ad9
begin

	first_image = zeros(Float64, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT
			first_image[X, row, column] = row_ratio(row)
			first_image[Y, row, column] = column_ratio(column)
			first_image[Z, row, column] = 0.25
		end
	end

	imshow(first_image)

end

# ╔═╡ 6737299b-ecc4-438a-ab5b-00748549a073
begin

	local color_img = Array{Color}(undef, IMAGE_HEIGHT, IMAGE_WIDTH)

	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT
		
			color_img[row, column] = Color{Float64}(
				row_ratio(row),
				column_ratio(column),
				0.25
			)
			
		end
	end
	
	imshow(color_as_matrix(color_img))
	
end

# ╔═╡ 47d6e544-38f5-4b66-bf9b-2cd5ec3dd23c
begin
	
	local color_img = Array{Color}(undef, IMAGE_HEIGHT, IMAGE_WIDTH)
	
	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT

			u = column_ratio(column)
			v = row_ratio(row)

			r = Ray(ORIGIN, find_pixel(u, v))
			
			color_img[row, column] =  ray_color(r)
		end
	end


	imshow(color_as_matrix(color_img))
		
end

# ╔═╡ c77a6d7e-74a1-4237-ba2b-c57fa2489cee
begin
	
	local color_img = Array{Color}(undef, IMAGE_HEIGHT, IMAGE_WIDTH)
	
	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT

			u = column_ratio(column)
			v = row_ratio(row)

			r = Ray(ORIGIN, find_pixel(u, v))
		
			color_img[row, column] =  ray_color(r, Sphere([0; 0; -1], 0.5))
		end
	end


	imshow(color_as_matrix(color_img))
		
end

# ╔═╡ 0de8faf1-5bb9-4efb-aa65-3be603e6ebd8
begin
	
	objects = Vector{Sphere}()
	push!(objects, Sphere([0; 0; -1], 0.5))
	push!(objects, Sphere([0; -100.5; -1], 100))

	local color_img = Array{Color}(undef, IMAGE_HEIGHT, IMAGE_WIDTH)
	
	for column = 1:IMAGE_WIDTH
		for row = 1:IMAGE_HEIGHT
			
			pixel_color = Color{Float64}(0, 0, 0)
			
			for sampling = 1:SAMPLES_PER_PIXEL
				u = column_ratio(column)
				v = row_ratio(row)
	
				r = Ray(ORIGIN, find_pixel(u, v))
			
				pixel_color = pixel_color + ray_color(r, objects, MAX_DEPTH)
				
			end

			color_img[row, column] = (1/SAMPLES_PER_PIXEL) * pixel_color
		end
	end

	imshow(color_as_matrix(color_img))
end

# ╔═╡ Cell order:
# ╟─ef4c7762-b0c3-11ec-1425-cd672582f162
# ╠═2cb0d923-bd4c-4216-b000-70873e41b8e8
# ╠═661dae0b-fab4-4c17-a7ed-de4439aaf132
# ╠═518200c4-dffa-44a5-ac3c-3dd1d1564ad9
# ╠═8cc10884-ad70-412c-89c0-f4957016a045
# ╠═6737299b-ecc4-438a-ab5b-00748549a073
# ╠═e54406fe-0be0-493a-bbd5-98e9f3146a8f
# ╠═34cb88ac-b5d1-4309-9edd-65e7127094ee
# ╟─47d6e544-38f5-4b66-bf9b-2cd5ec3dd23c
# ╠═c77a6d7e-74a1-4237-ba2b-c57fa2489cee
# ╠═0de8faf1-5bb9-4efb-aa65-3be603e6ebd8
# ╠═6e0d0040-b8ee-459e-b694-811ce03dd961
# ╟─30549a0b-bf46-4326-a49e-3ec704768f0b
# ╟─3e0c6fd2-0a60-4233-8cac-8cab8cc31466
# ╟─dbf8802b-1b70-424d-bd50-9f87d259db5c
# ╟─3a72f057-a8c2-4531-a2e6-c438157105d8
# ╟─5257c0a3-7cd6-4b4c-ac8b-123116c9f963
# ╟─97f17bbb-fa79-48cd-91a7-be72dca5820b
# ╠═1c72807c-a8e5-4416-b8e4-80b23d0f36ea
# ╠═cb6d5996-49c1-4d79-99d3-0713bbfaefd3
# ╠═4c1b0089-c914-4321-a63b-822373be067d
# ╟─71b959f9-9fc4-4e22-9f64-623ada64d111
# ╠═1b0094ce-71b8-4bb9-a2e1-53d716b70880
# ╟─517ff8a4-a3fc-43c9-adcc-4fcfc2eb449f
# ╟─a8d48d9d-821c-47f4-aaf2-931086cbfd1e
# ╟─c4940508-567f-494a-8612-a226236ac15d
# ╟─44f9dd29-6ced-4f79-925b-8c1d5eda8ac7
# ╟─0246705a-2f60-4ac9-8ba1-7cbba560a442
# ╟─86a6e1c7-c8f6-4a2e-ab1b-ce6c7b1684c6
# ╟─23244102-6c96-4d60-bb14-236b54e384d8
# ╟─39df6e80-0779-4937-b62f-d13e7b4a948d
# ╠═624d5e30-2558-4aa5-9820-76f6556984a5
